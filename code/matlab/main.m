% Main
clc
close all
clear all


dirs = dir('../../data/locations/')

% Location of base colormap image
base_colormap = '../../data/colormapBaseImg.png';

%% Gabor
% for all locations, for all images
for location_index = 1 : length(dirs)
    
    if(strcmp(dirs(location_index).name,'.') == 0 && ...
            strcmp(dirs(location_index).name,'..') == 0 && ...
            strcmp(dirs(location_index).name,'.DS_Store') == 0)
     
        dirs(location_index).name
        
        path = strcat( dirs(location_index).folder,"/", dirs(location_index).name);
        generateHighDimImageFeaturesFun(path,base_colormap);
        
    end
end

%% Isomap


generateIsomapLowDimTrainDataFun(dirs);
