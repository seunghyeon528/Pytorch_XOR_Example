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
![image](https://user-images.githubusercontent.com/77431192/140403022-32f1e11b-f95d-4544-b1a3-554ad00125bb.png)

Sigmoid function is used as activateion function.

## 1. Prepare Data
Datset will be saved as "data.pickle".
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
![image](https://user-images.githubusercontent.com/77431192/140401298-20e1d65a-8bd6-419c-af39-641e4e54af79.png)

