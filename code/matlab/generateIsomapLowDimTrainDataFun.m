function generateIsomapLowDimTrainDataFun(directories)

 
    %% 0. Specify a data structure containing all the raw x data you want to 
    %% explicitly project using Isomap (for your training set data and images only). 

    Nsamps = 375; %number of samples per image 

    %%create regular grid of patches to extract from each training image -- could replace with randomized?
    xind = floor(linspace(1,3072,Nsamps)); 

    Xbar = [];

    % For all the directories in Locations/
    
    N = length(directories);

    for folder_index = 1 : N
      
         if(strcmp(directories(folder_index).name,'.') == 0 && ...
            strcmp(directories(folder_index).name,'..') == 0 &&  ...
            strcmp(directories(folder_index).name,'.DS_Store') == 0 && ...
        strcmp(directories(folder_index).name,'ORCCA') == 0)

                Xall = load(char(strcat(directories(folder_index).folder,"/",directories(folder_index).name,"/Xall.mat")));
                Xall = Xall.Xall;
                
                
                [dimX,numPatches] = size(Xall.data{1});
                numImages = length(Xall.data);

                for ii=1:floor(numImages/2) %%only use first 4 images to extract training patches from
                    Xbar = [Xbar, Xall.data{ii}(:,xind)];
                end
         end

    end

    Xbar = Xbar'; %%flip for Isomap: row i is now 1x175 dim vector
    
    disp(size(Xbar));
    
    if ( max(size(Xbar)) < 150)
        error('Error: Xbar too small. Add more images or increase Nsamps');
    end
    save ../../data/training/XbarData.mat -mat Xbar

    %% 1. Run Isomap
    %Compute the Euclidean distance matrix for all data in Xbar:
    %the ith row and jth column of Dbar is the Euclidean distance between
    %Xbar(i,:) and Xbar(j,:). 
    Dbar = squareform(pdist(Xbar));
    
    options.dims = 8; %%Ramos paper suggests 8 is sufficient
    knearest = 10; %%8-10 seems good
    Ybar=Isomap(Dbar,'k',knearest,options);
   
    %%%Ybar.coords is cell array with coords for d-dim embeddings in Ybar.coords{d}.  
    %%%Ybar.index has the indices of points embedded (corresponding row of Xbar).
    
    save ../../data/training/XbarData.mat -mat Xbar
    save ../../data/training/YbarData.mat -mat Ybar