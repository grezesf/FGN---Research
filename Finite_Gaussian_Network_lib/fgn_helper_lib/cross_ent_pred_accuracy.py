# number of correct pred function
# to be used when target is a class number (eg 0-10)
# and output is a vector of length equal to the number of class
# so the class prediction is the argmax of the vector

# to be used by th.training()

def cross_ent_pred_accuracy(output, target):
    pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    correct = pred.eq(target.long().view_as(pred)).sum().item()
    return correct