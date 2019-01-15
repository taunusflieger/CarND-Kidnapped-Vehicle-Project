/*
 * particle_filter.cpp
 *
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

// number of particles
#define NUM_PARTICLES   500
#define EPS 1e-6



double  ParticleFilter::CalcMGP(double asso_x, double asso_y, double obs_x, double obs_y, double std_x, double std_y) {
	// calculate multivariate Gaussian probability
	double m1 = 1 / (2 * M_PI * std_x * std_y);
	double e1 = asso_x - obs_x;
	double e2 = asso_y - obs_y;
	double e3 = e1 * e1 / (2 * std_x * std_x);
	double e4 = e2 * e2 / (2 * std_y * std_y);

	return m1 * (exp(-(e3 + e4)));
}


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	if (is_initialized_)
		return;

	random_device dev;
    mt19937 gen(dev());
	
	num_particles_ = NUM_PARTICLES;
	particles_.resize(num_particles_);

	normal_distribution<double> N_x_init(0, std[0]);
	normal_distribution<double> N_y_init(0, std[1]);
	normal_distribution<double> N_theta_init(0, std[2]);

	for (int i = 0; i < num_particles_; i++) {
		double n_x;
		double n_y;
		double n_theta;

		n_x = x + N_x_init(gen);
		n_y = y + N_y_init(gen);
		n_theta = theta + N_theta_init(gen);

		particles_[i] = Particle {i, n_x, n_y, n_theta, 1.0};
	}
  is_initialized_ = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	random_device dev;
	mt19937 gen(dev());

	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);
	// predict all particles for the next time step
  	for (auto &particle : particles_) {
  
		double theta = particle.theta;
		double sin_theta = sin(theta);
		double cos_theta = cos(theta);

		// check for yaw rate equals zero
		if (fabs(yaw_rate) >= EPS) {
			particle.x += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin_theta);
			particle.y += velocity / yaw_rate * (cos_theta - cos(theta + yaw_rate * delta_t));
			particle.theta += yaw_rate * delta_t;
		} else {
			particle.x += velocity * delta_t * cos_theta;
			particle.y += velocity * delta_t * sin_theta;
		}
		
		particle.x += noise_x(gen);
		particle.y += noise_y(gen);
		particle.theta += noise_theta(gen);
	}
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations) {
	for (auto &obs : observations) {
		double min_dist = __DBL_MAX__;

		for (auto landmark : predicted) {
			// calculate euclidean distance between prediction and observation
			double calc_dist = dist(obs.x, obs.y, landmark.x, landmark.y);
			
			// found closer observation
			if (calc_dist < min_dist) {
				min_dist = calc_dist;
				obs.id = landmark.id;
			}
		}
  	}
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const vector<LandmarkObs> &observations, const Map &map_landmarks) {
	
	for (auto &particle : particles_) {

		// landmarks within sensor's range
		vector<LandmarkObs> predictions;

		// observations in MAP's coordinate system
		vector<LandmarkObs> transformation_obs;

		// variables needed for setAssociation
		vector<int> associations;
		vector<double> sense_x;
		vector<double> sense_y;

		//
    	// Step 1: Get all landmarks within the sensor range
    	//
		// only consider landmarks within sensor range
		for (auto map_landmark : map_landmarks.landmark_list) {

			double distance_to_landmark = dist(map_landmark.x_f, map_landmark.y_f, particle.x, particle.y);
			if (distance_to_landmark < sensor_range) {
				LandmarkObs landmark;
				landmark.id = map_landmark.id_i;
				landmark.x = map_landmark.x_f;
				landmark.y = map_landmark.y_f;
				predictions.push_back(landmark);
			}
		}

		//
    	// Step 2: Transform observations into map coordinates
    	//
		for (auto observation : observations) {

			// observation in MAP's coordinate system
			LandmarkObs observation_map;

			observation_map.id = observation.id;

			double cos_theta = cos(particle.theta);
			double sin_theta = sin(particle.theta);
			// |---------               Rotation                     ----------| Translation |
      		//                  X(object)*cos(theta) -   Y(object)*sin(theta) +    X(particle)
			observation_map.x = observation.x*cos_theta - observation.y*sin_theta + particle.x;
			observation_map.y = observation.x*sin_theta + observation.y*cos_theta + particle.y;

			transformation_obs.push_back(observation_map);
		}
		//
		// Step 3: Associate observations to landmarks within the sensor range
		//
		// find closest landmark to each observation
		// id of transformation_obs is set to id of closest landmark inside predictions
		dataAssociation(predictions, transformation_obs);

		particle.weight = 1.0; //re-initialize

		//l andmark measurement uncertainty std_x [m], std_y [m]
		double std_x = std_landmark[0];
		double std_y = std_landmark[1];

		//f or each observation...
		for (auto observation : transformation_obs) {

			int closest_landmark_index = 0;

			// find closest landmark
			for (int i = 0; i < predictions.size(); ++i) {
				if (predictions[i].id == observation.id)  {
					closest_landmark_index = i;
					break;
				}
			}

			// matching landmark in the map
			LandmarkObs association = predictions[closest_landmark_index];

			// push back association info to display it in the simulator
			associations.push_back(association.id);
			sense_x.push_back(observation.x);
			sense_y.push_back(observation.y);

			//
			// Step 4: Update the particles' weight
			//
			particle.weight *= CalcMGP(association.x, association.y, observation.x, observation.y, std_x, std_y);
		}

		SetAssociations(particle, associations, sense_x, sense_y);

  	} //end particles loop	
}


void ParticleFilter::resample() {
	// final vector with resampled particles
	vector<Particle> resampled_particles;
	vector<double> weights;
	
	random_device dev;
	mt19937 gen(dev());

	// get weights from current particles
	double weight_max = 0;
	for (auto p : particles_) {
		weights.push_back(p.weight);
		
		// find max weight
		if (p.weight > weight_max)
			weight_max = p.weight;
	}

	// initial index is initialized uniformly
	uniform_int_distribution<int> index_distribution(0, num_particles_-1);
	uniform_real_distribution<double> beta_distribution(0, 2 * weight_max);

    int index = index_distribution(gen);
	double beta = 0.0;

	// resampling wheel to replace particles with
	// low importance weight
	for (int i=0; i < num_particles_; ++i) {
		beta += beta_distribution(gen);

		while (weights[index] < beta) {
			beta -= weights[index];
			index = (index + 1) % num_particles_;
		}
		particles_[index].id = i;
		resampled_particles.push_back(particles_[index]);
	}
	particles_ = resampled_particles;
}


void ParticleFilter::SetAssociations(Particle& particle, const vector<int>& associations, 
                                     const vector<double>& sense_x, const vector<double>& sense_y)
{
    // particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

 	// Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();
 
    particle.associations = associations;
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
