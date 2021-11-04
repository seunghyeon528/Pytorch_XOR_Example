# Pytorch_XOR_Example
Simple neural_net acting as XOR classifier.

Following is XOR truth table. Output is 1 only when two inputs are different from each other. 

|input|output|
|------|---|
|00|0|
|01|1|
|10|1|
|11|0|



## 0. Model Structure
Following is the network representation. 
To put it simply, the model is 2 - layer fully connected neural-network. 

![image](https://user-images.githubusercontent.com/77431192/140403022-32f1e11b-f95d-4544-b1a3-554ad00125bb.png)

Sigmoid function is used as activateion function.
Refer to "model.py".

## 1. Prepare Data
Dataset will be saved as "data.pickle".
~~~
python generate_dataset.py
~~~

## 2. Train
After training, trained model will be saved as "model.pt"
~~~
python train.py
~~~

## 3. Test
laod pre trained model from file "model.pt", conduct some test.
~~~
python test.py
~~~
## 4. Test Results
![image](https://user-images.githubusercontent.com/77431192/140401320-50c345d5-54b0-486e-bd24-764f96d89d22.png)

Following is 3D graph of trained network, plotting each input number in x & y axis, output in z axis.
You can celarly confirm that trained network divide the region each containing (1,0), (0,1) and (0,0), (1,1).
This is kind of job that network comprising only one layer cannot do. 

![image](https://user-images.githubusercontent.com/77431192/140401298-20e1d65a-8bd6-419c-af39-641e4e54af79.png)

