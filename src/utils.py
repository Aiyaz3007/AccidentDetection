def isColab():
    try:
        import google.colab 
        return True
    except ModuleNotFoundError:
        return False
    
def collate_fn(self,batch):
    return tuple(zip(*batch))