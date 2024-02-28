# AI-in-Industry
Project for AI in Industry course: Improve usage of unsupervised data for definition of RUL-based maintenance policies. 

**We will run multiple experiments on our data to analyse the effect of domain knowledge injection via multiple approaches.**

- In the first 3 tasks we will experiment with different ratios and combinations of supervised and unsupervised data.
- In task 4 we will use a static regularizer to inject domain knowledge (RUL>0). 
- In task 5 we will use a lagrangian approach to dynamically maximize the weight of the regularizer.

To get reliable results, we will test these approaches on 30 different seeds and we will compute the mean value for the loss and its stanard deviation.

**Local execution of the project**
This project makes use of Docker: in order to run it locally, you need to have Docker and Docker Compose installed. After cloning the respository, start the container via Docker Compose, from the main directory of the project:

  docker-compose up
