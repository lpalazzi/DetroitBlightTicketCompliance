# DetroitBlightTicketCompliance

Final assignment for course 'Applied Machine Learning in Python' from the University of Michigan.

For this assignment, a function is created that trains a model to predict blight ticket compliance in Detroit using the provided [training set](train.csv). Using this model, the function returns a series of length 61001 with the data being the probability that each corresponding ticket from the [test set](test.csv) will be paid, and the index being the ticket_id.

The model makes predictions on the test set with a Area Under the ROC Curve (AUC) score of 0.761394498577.
