def get_dataloader_classes(dataloader):
    
    # returns an array with all the classes found in data loader:
    # future: optimize so doesnt go through whole dataset?
    
    classes = set()
    
    for _, batch_classes in dataloader:
        
        classes = classes.union(batch_classes.detach().cpu().numpy())
        
    return list(classes)
        