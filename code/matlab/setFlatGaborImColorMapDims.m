%setFlatGaborImColorMapDims.m
function [mapOut,imheight,imwidth,imnumchannels] = setFlatGaborImColorMapDims(imFileName)
Iraw = imread(imFileName);
[~,mapOut] = rgb2ind(Iraw,2^16); %Gabor conv only works on 2D inputs; need to use same map for ALL images
[imheight,imwidth,imnumchannels] = size(Iraw);