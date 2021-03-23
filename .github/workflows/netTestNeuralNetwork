# %load network.py

"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

# SQL Database Code
import mysql.connector

try:
    db = mysql.connector.connect(
        host="localhost",
        username="root",
        password="pass",
        database="networkdatabase"
    )
    mycursor = db.cursor(buffered=True)
    print("Successfully Connected to Database")
except:
    print("ERR: Could not connect to database")


# Change Graph Style
style.use("ggplot")

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        self.accuracy = []
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.eta = eta

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
                self.accuracy.append(round(self.evaluate(test_data)*100/n_test,2))

            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def plotAccuracy(self):
        plt.plot(self.accuracy)
        plt.ylabel('accuracy (%)')
        plt.xlabel('Number of Epochs (from 0)')
        plt.show()

    def findMax(self):
        self.epochPos = 0
        #must have arrayName[index].fieldName (since it is an array of records)
        self.maxi = self.accuracy[0]
        
        for i in range(1,self.epochs):

            if self.accuracy[i] > self.maxi:
                self.maxi = self.accuracy[i]
                self.epochPos = i

    def outputResults(self,networkName):
        with open("results.txt","a") as writefile:
            #modelID, modelName, numHidden, epochs, mini_batch_size, eta, modelID2, currentAccuracy, maxAccuracy, maxAccuracyEpoch, accuracyArray = x
            line = (networkName+","+ str(self.sizes[1]) + "," + str(self.epochs) +"," + str(self.mini_batch_size) + "," + str(self.eta) + "," + str(self.accuracy[len(self.accuracy)-1]) + "," + str(self.maxi) + "," + str(self.epochPos) + "," + str(self.accuracy) + "\n")
            writefile.write(line)

            #---- WRONG ORDER-------
            # #name, current accuracy, max accuracy, max accuracy epoch, accuracyArray, numHidden, epochs, mini_batch_size, eta
            # line = (networkName+","+ str(self.accuracy[len(self.accuracy)-1]) + "," + str(self.maxi) +"," + str(self.epochPos) + "," + str(self.accuracy) + "," + str(self.sizes[1]) + "," + str(self.epochs) + "," + str(self.mini_batch_size) + "," + str(self.eta) + "\n")
            # writefile.write(line)

        writefile.close()

    def writeModel(self,networkName):
        try:
            insertModel = "INSERT INTO Models (modelName, numHidden, epochs, mini_batch_size, eta) VALUES (%s,%s,%s,%s,%s)"
            mycursor.execute(insertModel,(networkName, str(self.sizes[1]), str(self.epochs), str(self.mini_batch_size), str(self.eta)))
            db.commit()

            last_id = mycursor.lastrowid
            # print(last_id)

            insertAccuracy = "INSERT INTO Accuracy (modelID, currentAccuracy, maxAccuracy, maxAccuracyEpoch, accuracyArray) VALUES (%s,%s,%s,%s,%s)"
            mycursor.execute(insertAccuracy,(last_id,str(self.accuracy[len(self.accuracy)-1]),str(self.maxi),str(self.epochPos),str(self.accuracy)))
            db.commit()
        except:
            print("ERROR: Unable to Insert")
    
    
    def databaseToFile(self):
        mycursor.execute("SELECT * FROM Models, Accuracy WHERE Models.ID = Accuracy.modelID")
        #print("Loop1")

        with open("results.txt","a") as writefile:
            #print("Loop2")
            for x in mycursor:
                #print("Loop3")
                modelID, modelName, numHidden, epochs, mini_batch_size, eta, modelID2, currentAccuracy, maxAccuracy, maxAccuracyEpoch, accuracyArray = x
                #name, current accuracy, max accuracy, max accuracy epoch, accuracyArray numHidden, epochs, mini_batch_size, eta
                line = (modelName+","+ str(numHidden) + "," + str(epochs) +"," + str(mini_batch_size) + "," + str(eta) + "," + str(currentAccuracy) + "," + str(maxAccuracy) + "," + str(maxAccuracyEpoch) + "," + str(accuracyArray) +"\n")
                # WRONG ORDER line = (modelName+","+ str(currentAccuracy) + "," + str(maxAccuracy) +"," + str(maxAccuracyEpoch) + "," + str(accuracyArray) + "," + str(numHidden) + "," + str(epochs) + "," + str(mini_batch_size) + "," + str(eta) +"\n")
                writefile.write(line)

        writefile.close()


    def getModel(self,queryID,queryName,queryAccuracy):
        searchModel = "SELECT modelID, modelName, numHidden, epochs, mini_batch_size, eta, currentAccuracy, maxAccuracy, maxAccuracyEpoch, accuracyArray FROM Models, Accuracy WHERE Models.ID = Accuracy.modelID AND modelID LIKE %s AND modelName LIKE %s AND maxAccuracy LIKE %s"
        print(queryAccuracy)
        #mycursor.execute(searchModel)
        mycursor.execute(searchModel, ("%" + queryID + "%","%" + queryName + "%", queryAccuracy + "%"))
        rows = mycursor.fetchall()
        rowCount = mycursor.rowcount
        # for x in mycursor:
        #     print(x)
        # print("Total Data Entries: ",rowCount)

        return rows,rowCount
    
    def deleteModel(self,ID):
        queryRecordExists = "SELECT COUNT(1) FROM Models WHERE ID = %s"
        mycursor.execute(queryRecordExists, (str(ID),))
        recordExists = mycursor.fetchall()
        print(recordExists)
        if recordExists == [(1,)]:
            deleteQueryAccuracy = "DELETE FROM Accuracy WHERE modelID = %s"
            deleteQueryModels = "DELETE FROM Models WHERE ID = %s"
            mycursor.execute(deleteQueryAccuracy, (str(ID),))
            mycursor.execute(deleteQueryModels, (str(ID),))
            db.commit()
            print("Record" ,ID, "Deleted")
            return True
        return False

    
    def showDatabase(self):
        # mycursor.execute("SELECT * FROM Models, Accuracy WHERE Models.ID = Accuracy.modelID") - This includes modelID twice
        mycursor.execute("SELECT modelID, modelName, numHidden, epochs, mini_batch_size, eta, currentAccuracy, maxAccuracy, maxAccuracyEpoch, accuracyArray FROM Models, Accuracy WHERE Models.ID = Accuracy.modelID")
        for x in mycursor:
            print(x)

    def readFileData(self):
        self.savedNetworks = []
        #self.savedNetworks = [[-1]*9 for i in range(10)] #This only works for 10 models
        with open('results.txt','r') as readFile:
            
            line = readFile.readline().rstrip("\n")
            counter = 0
            
            while line:
                self.savedNetworks.append([-1]*9)
                #modelName, numHidden, epochs, mini_batch_size, eta, currentAccuracy, maxAccuracy, maxAccuracyEpoch, accuracyArray = x
                
                items = line.split(",")

                self.savedNetworks[counter][0] = items[0] #name
                self.savedNetworks[counter][1] = int(items[1]) #numHidden
                self.savedNetworks[counter][2] = int(items[2]) #epochs
                self.savedNetworks[counter][3] = int(items[3]) #mini_batch_size
                self.savedNetworks[counter][4] = float(items[4]) #eta
                self.savedNetworks[counter][5] = float(items[5]) #currentAccuracy
                self.savedNetworks[counter][6] = float(items[6]) #maxAccuracy
                self.savedNetworks[counter][7] = int(items[7]) #maxAccuracyEpoch
                self.savedNetworks[counter][8] = str(items[8]) #accuracyArray
                
                #-----WRONG ORDER-------
                # self.savedNetworks[counter][0] = items[0] #name
                # self.savedNetworks[counter][1] = float(items[1]) #current accuracy
                # self.savedNetworks[counter][2] = float(items[2]) #max accuracy
                # self.savedNetworks[counter][3] = int(items[3]) #max accuracy epoch
                # self.savedNetworks[counter][3] = items[4] #accuracy array
                # self.savedNetworks[counter][4] = int(items[5]) #numHidden
                # self.savedNetworks[counter][5] = int(items[6]) #epochs
                # self.savedNetworks[counter][6] = int(items[7]) #mini_batch_size
                # self.savedNetworks[counter][7] = float(items[8]) #eta
                
                counter = counter + 1

                
                line = readFile.readline().rstrip("\n")
                
            #end while
        return self.savedNetworks

    def insertionSort(self,arr):
        for i in range(1,len(arr)): 
            currentValue = arr[i]
            pos = i
            while pos > 0 and arr[pos-1][6]<currentValue[6]:
                arr[pos] = arr[pos-1]
                pos -=1

            arr[pos] = currentValue
        #     print(i,pos)
        #     print(self.savedNetworks)
        # for i in range(len(self.savedNetworks)):
        #     print(self.savedNetworks[i])
        return arr

        #----------------- Sort using self.savedNetworks instead of arbitrary array ----------
        # for i in range(1,len(self.savedNetworks)): 
        #     currentValue = self.savedNetworks[i]
        #     pos = i
        #     while pos > 0 and self.savedNetworks[pos-1][6]<currentValue[1]:
        #         self.savedNetworks[pos] = self.savedNetworks[pos-1]
        #         pos -=1

        #     self.savedNetworks[pos] = currentValue
        # #     print(i,pos)
        # #     print(self.savedNetworks)
        # # for i in range(len(self.savedNetworks)):
        # #     print(self.savedNetworks[i])
        # return self.savedNetworks

    def binarySearch(self,sortedArr,target):
        #target = float(input("Enter the accuracy you want to search for"))
        lower = 0
        upper = len(sortedArr)-1
        pos = -1
        found = False

        while lower<=upper and not found:
            middle = (lower+upper)//2
            if sortedArr[middle][6] == target:
                found = True
                pos = middle
            elif sortedArr[middle][6] < target:
                upper = middle-1
            else:
                lower = middle+1
        
        if found:
            return sortedArr[pos]
        else:
            return -1
            # print("Here is the model:",self.savedNetworks[pos])

        #---------------Using self.savedNetworks instead of arbritrary sorted array --------
        # def binarySearch(self):
            # target = float(input("Enter the accuracy you want to search for"))
            # lower = 0
            # upper = len(self.savedNetworks)-1
            # pos = -1
            # found = False

            # while lower<=upper and not found:
            #     middle = (lower+upper)//2
            #     if self.savedNetworks[middle][1] == target:
            #         found = True
            #         pos = middle
            #     elif self.savedNetworks[middle][1] < target:
            #         upper = middle-1
            #     else:
            #         lower = middle+1
            
            # if found:
            #     return self.savedNetworks[pos]
            #     # print("Here is the model:",self.savedNetworks[pos])

    

    # Accessor Functions

    def getAccuracy(self):
        return self.accuracy

    def getMaxAccuracy(self):
        return self.maxi



#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

#Sigmoid Test plot
# z1 = np.arange(-5,5,0.1)
# plt.plot(z1,sigmoid(z1))
# plt.show()


