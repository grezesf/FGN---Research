def get_dataloader_bounds(dataloader):
    
    # given a data loader, finds the absolute min and max of the data
    # (future: get bounds per dimension? - optimize so only uses first batch, and doesn't go through batch twice?)
    
    mini = None
    maxi = None
    
    # go through the data
    for data, _ in dataloader:
        
        # find batch bounds
        batch_min = float(data.min().detach().cpu().numpy())
        batch_max = float(data.max().detach().cpu().numpy())
        
        # compare 
        if mini==None:
            mini = batch_min
        else:
            mini = min(mini,batch_min)
            
        if maxi==None:
            maxi = batch_max
        else:
            maxi = max(maxi,batch_max)

    # return tuple
    return (mini, maxi)