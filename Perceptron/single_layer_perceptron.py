import sys
import random
from matplotlib import pyplot as plt
import numpy as np

def plot(matrix,weights=None,title="Prediction Matrix"):

    if len(matrix[0])==3: # if 1D inputs, excluding bias and ys 
        fig,ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel("i1")
        ax.set_ylabel("Classifications")

        if weights!=None:
            y_min=-0.1
            y_max=1.1
            x_min=0.0
            x_max=1.1
            y_res=0.001
            x_res=0.001
            ys=np.arange(y_min,y_max,y_res)
            xs=np.arange(x_min,x_max,x_res)
            zs=[]
            for cur_y in np.arange(y_min,y_max,y_res):
                for cur_x in np.arange(x_min,x_max,x_res):
                    zs.append(predict([1.0,cur_x],weights))
            xs,ys=np.meshgrid(xs,ys)
            zs=np.array(zs)
            zs = zs.reshape(xs.shape)
            cp=plt.contourf(xs,ys,zs,levels=[-1,-0.0001,0,1],colors=('b','r'),alpha=0.1)
        
        c1_data=[[],[]]
        c0_data=[[],[]]

        for i in range(len(matrix)):
            cur_i1 = matrix[i][1]
            cur_y  = matrix[i][-1]

            if cur_y==1:
                c1_data[0].append(cur_i1)
                c1_data[1].append(1.0)
            else:
                c0_data[0].append(cur_i1)
                c0_data[1].append(0.0)

        plt.xticks(np.arange(x_min,x_max,0.1))
        plt.yticks(np.arange(y_min,y_max,0.1))
        plt.xlim(0,1.05)
        plt.ylim(-0.05,1.05)

        c0s = plt.scatter(c0_data[0],c0_data[1],s=40.0,c='r',label='Class 0')
        c1s = plt.scatter(c1_data[0],c1_data[1],s=40.0,c='b',label='Class 1')

        plt.legend(fontsize=10,loc=1)
        plt.show()
        return

    if len(matrix[0])==4: # if 2D inputs, excluding bias and ys
        fig,ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel("i1")
        ax.set_ylabel("i2")

        if weights!=None:
            map_min=0.0
            map_max=1.1
            y_res=0.001
            x_res=0.001
            ys=np.arange(map_min,map_max,y_res)
            xs=np.arange(map_min,map_max,x_res)
            zs=[]
            for cur_y in np.arange(map_min,map_max,y_res):
                for cur_x in np.arange(map_min,map_max,x_res):
                    zs.append(predict([1.0,cur_x,cur_y],weights))
            xs,ys=np.meshgrid(xs,ys)
            zs=np.array(zs)
            zs = zs.reshape(xs.shape)
            cp=plt.contourf(xs,ys,zs,levels=[-1,-0.0001,0,1],colors=('b','r'),alpha=0.1)

        c1_data=[[],[]]
        c0_data=[[],[]]
        for i in range(len(matrix)):
            cur_i1 = matrix[i][1]
            cur_i2 = matrix[i][2]
            cur_y  = matrix[i][-1]
            if cur_y==1:
                c1_data[0].append(cur_i1)
                c1_data[1].append(cur_i2)
            else:
                c0_data[0].append(cur_i1)
                c0_data[1].append(cur_i2)

        plt.xticks(np.arange(0.0,1.1,0.1))
        plt.yticks(np.arange(0.0,1.1,0.1))
        plt.xlim(0,1.05)
        plt.ylim(0,1.05)

        c0s = plt.scatter(c0_data[0],c0_data[1],s=40.0,c='r',label='Class 0')
        c1s = plt.scatter(c1_data[0],c1_data[1],s=40.0,c='b',label='Class 1')

        plt.legend(fontsize=10,loc=1)
        plt.show()
        return
    
    print("Matrix dimensions not covered.")
    
# gets the predicted output based on current weights
def predict(features, weights):
    threshold = 0.0
    activation = 0.0 # total sum
    for feature,weight in zip(features, weights):
        activation += feature * weight
    return 1.0 if activation >= threshold else 0.0

# calculates prediction accuracy
def accuracy(matrix, weights):
    correct = 0
    preds = []
    for features in matrix:
        pred = predict(features[:-1], weights)
        preds.append(pred)
        if pred == features[-1]: correct += 1.0
    print("predictions:", preds)
    # return overall prediction accuracy
    return correct / len(matrix)

#train the perceptron on the data found on the matrix
#trained weights are returned at the end of the function
def train_weights(matrix, weights, nb_epoch=10, l_rate=1.0,
                  do_plot=False, stop_early=True, verbose=True):
    #iterate for the number of epochs requested
    for epoch in range(nb_epoch):
        current_accuracy = accuracy(matrix, weights)
        #print out information
        print("\nEpoch %d \nWeights" %epoch, weights)
        print("Accuracy: ", current_accuracy)
        
        #check if we finished training
        if current_accuracy == 1.0 and stop_early: break
        
        #check if we should plot the current results
        if do_plot: plot(matrix, weights, title="Epoch %d"%epoch)
            
        #iterate over each training input
        for i in range(len(matrix)):
            #calculate prediction
            prediction = predict(matrix[i][:-1], weights)
            error = matrix[i][-1] - prediction
            
            if verbose:
                print("We are training on data at index %d..."%i)
            
            #iterate over each weight and update it
            for j in range(len(weights)):
                if verbose: sys.stdout.write("\tWeight[%d]: %0.5f --> "%(j, weights[j]))
                weights[j] += l_rate * error * matrix[i][j]
                if verbose: sys.stdout.write("%0.5f\n"%weights[j])
    
    #plot out the final results
    plot(matrix, weights, title="Final Epoch")
    return weights

def main():
    
    part_A = True
    data = []
    
    if part_A:
        data = [#  x0(bias)    x1       x2       y  
                    [1.00,    0.08,    0.72,    1.0],
                    [1.00,    0.10,    1.00,    0.0],
                    [1.00,    0.26,    0.58,    1.0],
                    [1.00,    0.35,    0.95,    0.0],
                    [1.00,    0.45,    0.15,    1.0],
                    [1.00,    0.60,    0.30,    1.0],
                    [1.00,    0.70,    0.65,    0.0],
                    [1.00,    0.92,    0.45,    0.0]
                ]
    else:
        data = [#  x0(bias)    x1       y     
                    [1.00,    0.08,    1.0],
                    [1.00,    0.10,    0.0],
                    [1.00,    0.26,    1.0],
                    [1.00,    0.35,    0.0],
                    [1.00,    0.45,    1.0],
                    [1.00,    0.60,    1.0],
                    [1.00,    0.70,    0.0],
                    [1.00,    0.92,    0.0]
               ]
        
    weights = [random.uniform(-1, 1) for i in range(len(data[0]) - 1)]
    
    train_weights(data, weights=weights, nb_epoch=1000,
                   l_rate=1.0, do_plot=True, stop_early=True, verbose=False)
        

if __name__ == '__main__':
    main()
