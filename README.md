## Objective
I wrote this simplistic neural network in order to learn on a deeper level how
these systems work, and how parameters to these systems (such as learning rate,
training data, and the number of iterations) affect the outcome.

For this network, the task I assigned was simple: given a list of K integers,
return the most common sign. For example:
```
When K = 5,
{ -0.23098, 0.128712, 0.597823, -0.2344, -0.50585 }

Would return -1
```

I accomplished this with a simple neural network with an input layer of 5 nodes,
and hidden layer with 5 nodes and an output layer with a single node, along with
an activation function which just outputs the sign of the computed sum.

## Findings
In my use case, I found that training for around 1 million iterations provided
the best results, with an average final error rate of about 0.3 (which is still
bad, but a drastic improvement over the initial error rate of 1.3). I also
tried a bunch of different values for the learning rate, but a value of around
0.00001 - 0.000001 worked the best.
