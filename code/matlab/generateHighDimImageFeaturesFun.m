
function generateHighDimImageFeaturesFun(path)
    %generateHighDimImageFeatures.m
    %%Read instructions in project description.
    %%You will need to change entries where you see the '***' comments.
    %%Output data will be saved to the folder specified in imLocationDir.

    gwaveLength = 2 ; %Gabor kernel wavelength, in pixels per cycle; must pick to be <=W/2 to avoid aliasing (W = min image width) 
    gorients = [0 45 90 135];
    gaborBank = gabor(gwaveLength,gorients);
    patchSize = [5 5]; %%%(Note: use of 5x5 patches limits Gabor Filter wavelength to 2 pix/cycle)

    imLocationDir = char(path); %% *** change this folder location

    disp([imLocationDir,'/*.png']);

    %% Step 0. Set base color map for ALL images to do Gabor conv
    [map,imheight,imwidth,imnumchannels] = ...
        setFlatGaborImColorMapDims('../../data/locations/ORCCA/orcca_0.png'); %% *** set the default base image 

    %% Step 1. Generate/store sample training data for each image in this directory
    %%%---> Be sure to change the directory where this script is looking:
    %%%     it will look for all .png files wherever you point it to, 
    %%%     and then generate raw feature data for those .png files.


    dstrct = struct2cell(dir([imLocationDir,'/*.png']));
    
    D = dir([imLocationDir, '/*.png']);
    num_images = length(D(not([D.isdir])));

    
    
    imfileNames = dstrct(1,1:num_images);
    imfileFolders = dstrct(2,1:num_images);

    numFiles = length(imfileNames);
    
    Xall.data = cell(numFiles,1);
    Xall.imfiles = cell(length(numFiles),1);

    Xall.map = map;
    Xall.imheight = imheight;
    Xall.imwidth = imheight;
    Xall.imnumchannels = imnumchannels;
    Xall.gaborWaveLength = gwaveLength;
    Xall.gaborOrients = gorients;
    Xall.patchSize = patchSize; 

    for ii=1:numFiles
        imfile = char(strcat(imfileFolders{ii},"/",imfileNames{ii}));
        imfile
        Iraw2 = imread(imfile);
        I2flat = rgb2ind(Iraw2,map); %%Gabor-friendly color-flattened image

        gaborMagii = imgaborfilt(I2flat,gaborBank);

        %%Process image patches and stack results
        funG = @(block_struct) imgaborfilt(block_struct.data, gaborBank);
        I3 = blockproc(I2flat,patchSize,funG);

        %%use mat2cell to split up patches for each Gabor filter output
        clear I4c1 I4c2 I4c3 I4c4 
        I4c1 = mat2cell(I3(:,:,1),patchSize(1)*ones(1,imheight/patchSize(1)),patchSize(2)*ones(1,imwidth/patchSize(2)));
        I4c2 = mat2cell(I3(:,:,2),patchSize(1)*ones(1,imheight/patchSize(1)),patchSize(2)*ones(1,imwidth/patchSize(2)));
        I4c3 = mat2cell(I3(:,:,3),patchSize(1)*ones(1,imheight/patchSize(1)),patchSize(2)*ones(1,imwidth/patchSize(2)));
        I4c4 = mat2cell(I3(:,:,4),patchSize(1)*ones(1,imheight/patchSize(1)),patchSize(2)*ones(1,imwidth/patchSize(2)));

        %%cell elements in the same spot of image should then get stacked with
        %%original raw RGB data to form patch training vectors from image
        clear I5r I5g I5b
        I5r = mat2cell(Iraw2(:,:,1),patchSize(1)*ones(1,imheight/patchSize(1)),patchSize(2)*ones(1,imwidth/patchSize(2)));
        I5g = mat2cell(Iraw2(:,:,2),patchSize(1)*ones(1,imheight/patchSize(1)),patchSize(2)*ones(1,imwidth/patchSize(2)));
        I5b = mat2cell(Iraw2(:,:,3),patchSize(1)*ones(1,imheight/patchSize(1)),patchSize(2)*ones(1,imwidth/patchSize(2)));

        lxvec = prod(patchSize)*3 + prod(patchSize)*length(gorients);
        numxvecs = numel(I4c1);
        X = zeros(lxvec,numxvecs);
        for vv=1:numxvecs
            X(:,vv) = [double(I5r{vv}(:));
                double(I5g{vv}(:));
                double(I5b{vv}(:));
                double(I4c1{vv}(:));
                I4c2{vv}(:);
                I4c3{vv}(:);
                I4c4{vv}(:)];
        end
        %%Store image data in structure
        Xall.data{ii} = X;
        Xall.imfiles{ii} = imfile;
    end
    %%save
     save([imLocationDir,'/Xall.mat'], '-mat', 'Xall');
end