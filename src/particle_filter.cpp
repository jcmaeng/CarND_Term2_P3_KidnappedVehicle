/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static int NUM_PARTICLES_MAX = 100;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = NUM_PARTICLES_MAX;
    default_random_engine gen;

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    
    //intialising weights and particles
    weights = vector<double>(num_particles);

    for (int i = 0; i < num_particles; i++) {
        double sample_x = dist_x(gen);
        double sample_y = dist_y(gen);
        double sample_theta = dist_theta(gen);

        Particle p;

        p.x = sample_x;
        p.y = sample_y;
        p.theta = sample_theta;
        p.weight = 1.0;
        p.id = i;

        particles.push_back(p);

        weights[i] = 1.0;
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    for (int i = 0; i < particles.size(); i++) {
        Particle &p = particles[i];
        if (fabs(yaw_rate) < 1e-10) {  // to avoid division by zero
            p.x += velocity * delta_t * cos(p.theta);
            p.y += velocity * delta_t * sin(p.theta);
        } else {
            p.x += velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
            p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
            p.theta += yaw_rate * delta_t;
        }

        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_theta(gen);        
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for (int i = 0; i < observations.size(); i++) {
        double dist_min = numeric_limits<double>::max();
        LandmarkObs &ob = observations[i];
        for (int j = 0; j < predicted.size(); j++) {
            LandmarkObs pr = predicted[j];
            double distance = dist(ob.x, ob.y, pr.x, pr.y);
            if (dist_min > distance) {
                dist_min = distance;
                ob.id = pr.id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    weights.clear();
    vector<Map::single_landmark_s> lm_list = map_landmarks.landmark_list;

    for (int i = 0; i < particles.size(); i++) {
        Particle &p = particles[i];

        // choose landmarks within square area around particle
        // cout << "---------------------select obs--------------------" << endl;
        vector<LandmarkObs> selected_obs;
        for (int j = 0; j < lm_list.size(); j++) {
            Map::single_landmark_s lm = lm_list[j];
            if ((fabs(lm.x_f - p.x) < sensor_range) && (fabs(lm.y_f - p.y) < sensor_range)) {
                selected_obs.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
                // cout << lm.id_i << ":" << lm.x_f << ":" << lm.y_f << endl;
            }
        }

        // transform observed landmarks
        // cout << "--------------------transform----------------------" << endl;
        // cout << "# observation: " << observations.size() << endl;
        vector<LandmarkObs> transformed_obs;
        for (int k = 0; k < observations.size(); k++) {
            LandmarkObs ob = observations[k];
            LandmarkObs tmp;

            tmp.x = ob.x * cos(p.theta) - ob.y * sin(p.theta) + p.x;
            tmp.y = ob.x * sin(p.theta) + ob.y * cos(p.theta) + p.y;
            tmp.id = k;
            // cout << tmp.id << ":" << tmp.x << ":" << tmp.y << endl;
            transformed_obs.push_back(tmp);
        }

        dataAssociation(selected_obs, transformed_obs);

        p.weight = 1.0;
        // cout << "# transform: " << transformed_obs.size() << endl;
        for (int m = 1; m < transformed_obs.size(); m++) {
            LandmarkObs tob = transformed_obs[m];
            Map::single_landmark_s mlm = lm_list.at(tob.id-1);

            // compute weight using multi-variate gaussian distribution
            double x_diff = pow(tob.x - mlm.x_f, 2) / (2 * pow(std_landmark[0], 2));
            double y_diff = pow(tob.y - mlm.y_f, 2) / (2 * pow(std_landmark[1], 2));
            double w = exp(-(x_diff + y_diff)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
            p.weight *= w;
        }

        weights.push_back(p.weight);
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen;
    discrete_distribution<int> dist(weights.begin(), weights.end());

    vector<Particle> resample_ps;

    for (int i = 0; i < num_particles; i++) {
        resample_ps.push_back(particles[dist(gen)]);
    }

    particles = resample_ps;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
