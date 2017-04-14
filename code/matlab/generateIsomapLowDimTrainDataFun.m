function generateIsomapLowDimTrainDataFun(directories)

 
    %% 0. Specify a data structure containing all the raw x data you want to 
    %% explicitly project using Isomap (for your training set data and images only). 

    Nsamps = 1400; %number of samples per image 

    %%create regular grid of patches to extract from each training image -- could replace with randomized?
    xind = floor(linspace(1,3072,Nsamps)); 

    Xbar_R = [];
    Xbar_Q = [];

   

    % For all the directories in Locations/
    N = length(directories);
    
    valid_location_index = [];
    locations_to_indices_R = struct;
    locations_to_indices_Q = struct;

    
    for location_index = 1 : N
      
         if(strcmp(directories(location_index).name,'.') == 0 && ...
            strcmp(directories(location_index).name,'..') == 0 &&  ...
            strcmp(directories(location_index).name,'.DS_Store') == 0)

               valid_location_index = [valid_location_index,location_index];
               
               locations_to_indices_R.(directories(location_index).name) = [];
               locations_to_indices_Q.(directories(location_index).name) = [];
               
         end

    end
    
    
    for location_index = valid_location_index
        
            Xall = load(char(strcat(directories(location_index).folder,"/",directories(location_index).name,"/Xall.mat")));
            Xall = Xall.Xall;

            [dimX,numPatches] = size(Xall.data{1});
            numImages = length(Xall.data);
            used_images = floor(numImages/2);
            
            locations_to_indices_R.(directories(location_index).name) = zeros(length(xind),used_images);
            locations_to_indices_Q.(directories(location_index).name) = zeros(length(xind),numImages - used_images);

            for ii=1:used_images %%only use first 4 images to extract training patches from
                   
                % Indices are formatted in a python fashion (starting at 0)
                
                new_indices = [length(Xbar_R):length(Xall.data{ii}(:,xind)) - 1 + length(Xbar_R)];
                
                Xbar_R = [Xbar_R, Xall.data{ii}(:,xind)];
                
              
                locations_to_indices_R.(directories(location_index).name)(:,ii) = new_indices';
                
            end
            
            for ii = used_images + 1:numImages
                   
                % Indices are formatted in a python fashion (starting at 0)
                
                new_indices = [length(Xbar_Q):length(Xall.data{ii}(:,xind)) - 1 + length(Xbar_Q)];
                
                Xbar_Q = [Xbar_Q, Xall.data{ii}(:,xind)];
                
              
                locations_to_indices_Q.(directories(location_index).name)(:,ii) = new_indices';
                
            end
                
                

    end

    Xbar_R = Xbar_R'; %%flip for Isomap: row i is now 1x175 dim vector
    
    disp(size(Xbar_R));
    
    if ( max(size(Xbar_R)) < 150)
        error('Error: Xbar too small. Add more images or increase Nsamps');
    end

    %% 1. Run Isomap
    %Compute the Euclidean distance matrix for all data in Xbar:
    %the ith row and jth column of Dbar is the Euclidean distance between
    %Xbar(i,:) and Xbar(j,:). 
    Dbar = squareform(pdist(Xbar_R));
    
    options.dims = 8; %%Ramos paper suggests 8 is sufficient
    knearest = 10; %%8-10 seems good
    Ybar_R=Isomap(Dbar,'k',knearest,options);
   
    %%%Ybar.coords is cell array with coords for d-dim embeddings in Ybar.coords{d}.  
    %%%Ybar.index has the indices of points embedded (corresponding row of Xbar).
    
    save ../../data/training/locations_to_indices_R.mat -mat locations_to_indices_R
    save ../../data/training/locations_to_indices_Q.mat -mat locations_to_indices_Q

    save ../../data/training/Ybar_R.mat -mat Ybar_R
    save ../../data/training/Xbar_R.mat -mat Xbar_R
    save ../../data/training/Xbar_Q.mat -mat Xbar_Q
