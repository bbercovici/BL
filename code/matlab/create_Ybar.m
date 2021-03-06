% Creates the Ybar vector
clc 
close all
clear all

Y_bar_R =  importdata('../../data/training/Ybar_R.txt');
Y_bar_Q =  importdata('../../data/training/Ybar_Q.txt');

location_to_Q_indices = load('../../data/training/location_to_Q_indices.mat');
location_to_R_indices = load('../../data/training/location_to_R_indices.mat');

location_to_Q_indices = location_to_Q_indices.('location_to_Q_indices');
location_to_R_indices = location_to_R_indices.('location_to_R_indices');

locations = fieldnames(location_to_Q_indices);

Ybar = struct;

% For each location
for loc_index = 1 : numel(locations)
  
  imgs = fieldnames(location_to_Q_indices.(locations{loc_index}));
  
  Ybar.(locations{loc_index}) = [];
  
  % For each image
  for img_index = 1 : numel(imgs)
      
      Ybar_Q_indices = 1 + location_to_Q_indices.(locations{loc_index}).(imgs{img_index});
      Ybar_R_indices = 1 + location_to_R_indices.(locations{loc_index}).(imgs{img_index});

      Ybar.(locations{loc_index}) = [ Ybar.(locations{loc_index}), Y_bar_Q(:,Ybar_Q_indices)];
      Ybar.(locations{loc_index}) = [ Ybar.(locations{loc_index}), Y_bar_R(:,Ybar_R_indices)];
           
  end

 
end

save ../../data/training/Ybar.mat -mat Ybar
