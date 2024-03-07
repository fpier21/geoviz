from cv2 import VideoWriter, VideoWriter_fourcc, destroyAllWindows
from math import ceil, floor
import numpy as np
from matplotlib import pyplot as plt, cm 
from moviepy.video.io.bindings import mplfig_to_npimage
from mlxtend.plotting import plot_decision_regions
import torch
from copy import deepcopy
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D

colormap = cm.get_cmap("rainbow")


def plot_grid_func(
    n_lines: int,
    line_points: int,
    ax,
    map_func = None,
    starting_grid = None, 
    dim = 2, 
    x_min = None, 
    x_max = None, 
    y_min = None, 
    y_max = None
):
    """
    Plot a transformation of a regular grid.

    :param xmin: Minimum x value
    :param xmax: Maximum x value
    :param ymin: Minimum y value
    :param ymax: Maximum y value
    :param n_lines: Number of lines per axis
    :param line_points: Number of points per line
    :param map_func: Function to map the grid points to new coordinates
    """
    # List for gathering the lines into.
    lines = []

    if map_func is None: 
        map_func = nn.Identity()

    
    if starting_grid is None:
        # Iterate over horizontal lines.
        for y in np.linspace(y_min, y_max, n_lines): 
            lines.append(np.array([[x,y] for x in np.linspace(x_min, x_max, line_points)]))  
        
        # Iterate over vertical lines.
        for x in np.linspace(x_min, x_max, n_lines):
            lines.append(np.array([[x,y] for y in np.linspace(y_min, y_max, line_points)]))

    else: 
        for line in starting_grid: 
            with torch.no_grad():
                transformed_line = map_func(torch.Tensor(line)).numpy()
            lines.append(transformed_line)

    step = int(len(lines)/(n_lines*2)) ## adapt this if we allow starting 3d data with corresponding 3d axis grid
    to_keep = [lines[floor(len(lines)/2)], lines[ceil(len(lines)/2)], lines[-1]]

    for i, line in enumerate(lines[::step] + to_keep):
        i = i*step
        if dim ==2: 
            p = i / (len(lines) - 1)  # Normalize to 0-1.
            # Get the line color from the colormap.
            ax.plot(line[:,0], line[:,1], color=colormap(p), linewidth = 0.7)
        elif dim ==3: 
            p = i / (len(lines) - 1) 
            ax.plot3D(line[:,0], line[:,1], line[:,2],color=colormap(p), linewidth = 0.7)
        else: 
            raise Exception("")

    return lines

class geo_viz: 

    def __init__(self, X_train, y_train, model): 
        """
        X_train, X_label must be array-like
        """

        self.transformations = model.transformations
        self.n_epochs = len(self.transformations)

        self.X_train, self.y_train = self.validateXy(X_train,y_train)
        
        for idx, (layer, n_neurons) in enumerate(model.neurons_count().items()): 
            if n_neurons > 3: 
                raise Exception(f"Layer n. {idx} ({layer}) has more than 3 neurons. Impossible to plot.") # Implement T-SNE or something like this? 
        
        self.n_layers = len(model.neurons_count())

    @staticmethod
    def validateXy(X,y):
         if not isinstance(X,np.ndarray): ## facultative really
             try: 
                X = np.array(X)
             except: 
                  raise Exception("Impossible to cast")
         ### ecccc

         return np.array(X),np.array(y)
    
    def transform(self, input):
        
        pass 

    def plot(self, epoch, figsize = None, annotate = None, plot_grid = False, border = 0.05):
        """ 
        plot for one epoch all the transformations
        """

        self.epoch = epoch
        self.annotate = annotate
        self.plot_grid = plot_grid

        classes = np.unique(self.y_train)
        markers = ['o','<'] ## extend this list

        dim = self.X_train.shape[1] ## put this in a better place

        ## Initialize matplotlib fig based on layer number
        fig = plt.figure(figsize = figsize, num=epoch, clear = True)
        nrows = self.n_layers
        ncols = 2 # taking into account activations output

        # Input points 
        ax = fig.add_subplot(nrows,ncols,1)

        for i,class_ in enumerate(classes):   
            if dim == 2: 
                ax.plot(self.X_train[self.y_train==class_, 0], self.X_train[self.y_train==class_,1], markers[i], alpha = 0.75, markersize= 5) ## adapt with input vector y_label
            elif dim == 3: 
                pass
            else: 
                raise Exception("")
                ## ax.3Dplot ecc ecc

        if self.annotate: 
                step = int(len(self.X_train)*self.annotate) + 1 ## improve this
                for i,txt in enumerate(range(self.X_train.shape[0])[:step:]): 
                    if dim == 2 : 
                        ax.annotate(txt, (self.X_train[i, 0], self.X_train[i,1]))
                    elif dim == 3: 
                        pass
                    else: 
                        raise Exception("")

                    
        if self.plot_grid: ## adapt plot grid to use pytorch model.transformations functions
                    
                    x_min_, x_max_ = np.min(self.X_train[:,0]), np.max(self.X_train[:,0])
                    y_min_, y_max_ = np.min(self.X_train[:,1]), np.max(self.X_train[:,1])
                    

                    if border: 
                        x_min = x_min_ - abs(x_max_-x_min_)*border
                        x_max = x_max_ + abs(x_max_-x_min_)*border
                        y_min = y_min_ - abs(y_max_-y_min_)*border
                        y_max = y_max_ + abs(y_max_-y_min_)*border
                    
                    x_min = x_min_
                    x_max = x_max_
                    y_min = y_min_
                    y_max = y_max_
                    

                    lines_identity = plot_grid_func(10,10,ax, map_func = None, x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max) 
                    tmp_grid = lines_identity

                
        ax.set_title('Input Points')

        ## empty plot 
        ax = fig.add_subplot(nrows,ncols,2)

        plot_decision_regions(self.X_train, self.y_train.astype(int), self)
        ax.set_title('Decision boundary')
      
        
        # Layers and activations
        
        output = torch.Tensor(self.X_train)
        
        #for id, (key,output_2) in enumerate(self.outputs[epoch].items()): 
        for id, (key,transformation) in enumerate(self.transformations[epoch].items()): 
            
            output = transformation(output).data

            n_img = id+3
            dim = output.shape[1] ## output neurons

            if dim ==1 or dim == 2: 
                ax = fig.add_subplot(nrows,ncols,n_img)
            elif dim == 3: 
                ax = fig.add_subplot(nrows,ncols,n_img, projection='3d')

            for i,class_ in enumerate(classes):  
                
                out = output.numpy()[self.y_train == class_, :]
                #print(out.shape)

                if dim == 1: 
                            #ax.set_xscale("log")
                            
                            ax.plot(out[:,0],np.zeros_like(out) + 0, markers[i], alpha = 0.75, markersize= 5) ## adapt with input vector y_label            
                            #ax.get_xaxis().set_ticks([])
                elif dim == 2:  
                            ax.plot(out[:,0], out[:,1], markers[i], alpha = 0.75, markersize= 5) ## adapt with input vector y_label
                elif dim == 3: 
                            ax.scatter(out[:,0], out[:,1], out[:,2], marker= markers[i], alpha = 0.75, s= 5) 
                else: 
                    raise Exception(f"dim {dim} not allowed")
            

            if self.annotate: 
                step = int(len(self.X_train)*self.annotate) + 1 ## improve this
                for i,txt in enumerate(range(self.X_train.shape[0])[:step:]): 
                    if dim == 1 : 
                        ax.annotate(txt, (output.numpy()[i],0))
                    elif dim == 2 : 
                        ax.annotate(txt, (output.numpy()[i, 0], output.numpy()[i,1]))
                    elif dim == 3: 
                        ax.text(output.numpy()[i, 0], output.numpy()[i, 1],output.numpy()[i, 2],txt, zdir='x', fontsize = 'xx-small')
                    else: 
                        raise Exception("")
                    
            if self.plot_grid:

                if dim == 1: 
                            #ax.plot(out[:,0],np.zeros_like(out) + 0, markers[i], alpha = 0.75, markersize= 5) ## adapt with input vector y_label
                     pass            
                elif dim == 2:       
                    tmp_grid = plot_grid_func(10,10,ax, map_func = transformation, starting_grid=tmp_grid)
                elif dim == 3: 
                    tmp_grid = plot_grid_func(10,10,ax, map_func = transformation, starting_grid=tmp_grid, dim = 3)
                else: 
                    raise Exception(f"dim {dim} not allowed")
                #lines_identity = plot_grid(x_min,x_max,y_min,y_max,10,10,ax, map_func = None)

            ax.set_title(key)
            fig.suptitle(f'Epoch {epoch}')          
            fig.tight_layout(rect=[0, 0, 1, 0.96])

        return fig 


    def make_video(self, fps = 1, title = 'video_out.avi', step = None, figsize = None, annotate = False, plot_grid = False): 

        if step is None: 
            step = int(self.n_epochs/10)

        for epoch in range(self.n_epochs):
          if (epoch%step == 0 or epoch + 1 == self.n_epochs): 
            img= self.plot(epoch, figsize = figsize, annotate=annotate, plot_grid=plot_grid)
            if epoch == 0: 
                 figsize_int = img.get_size_inches()*img.dpi
                 out = VideoWriter(title,VideoWriter_fourcc(*'DIVX'), fps, tuple(figsize_int.astype(int)))
            out.write(mplfig_to_npimage(img))

            #closing figure
            img.clf()
            plt.close(img)

        out.release()
        destroyAllWindows()
        
        return 

    def predict(self,x):
        """
        mimic pytorch nn.Module method
        """
        
        x = torch.tensor(x, dtype = torch.float32)        
        for trasf in self.transformations[self.epoch].values():
            x = trasf(x)

        pred = x[:,0]
        return (pred>=0.5).float()


class VizModule(nn.Module):
    def __init__(self): 
        super().__init__()
        self.transformations = {}
        
    def register_transformations(self,epoch):      
        self.transformations[epoch]={}

        with torch.no_grad():      
                for layer, module in self.named_children(): 
                    self.transformations[epoch][layer] = deepcopy(module) 

    def viz_update(self,epoch): 
        """
        alias for register_transformations
        """
        self.register_transformations(epoch)
                    
    def neurons_count(self): 
        res = {}
        for id, (layer,module) in enumerate(self.named_children()):
            if isinstance(module, nn.Linear):
                if id == 0: 
                    res['input'] = module.in_features
                res[layer] = module.out_features
                 
        return res