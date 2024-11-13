import torch
from sklearn.model_selection import train_test_split

class FinData(torch.utils.data.Dataset):
    def __init__(self, x, y, tokenizer, maxlen, device='cpu'):self.x,self.y,self.tokenizer,self.maxlen= x,y,tokenizer,maxlen
    def __len__(self):return len(self.x)
    def __getitem__(self, idx):
        # tokenizer the input sequence
        inputs = self.tokenizer(self.x[idx], return_tensors="pt", padding="max_length", truncation=True, max_length=self.maxlen)
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        label = torch.tensor(self.y[idx], dtype=torch.long)
        # return input ids, attn mask, and labels
        return {"input_ids": input_ids, "attention_mask": attention_mask}, label


class GET_DLS:
    def __init__(self, data, tokenizer, maxlen, tsz, random_state):
        # initialize the arguments to the class
        self.data, self.tokenizer, self.maxlen, self.tsz, self.random_state = data, tokenizer,  maxlen, tsz, random_state
    def get_dls(self, batch_size):
        # split training and testing data
        X_train, X_test, y_train, y_test = train_test_split(self.data['news'], self.data['sentiment'], test_size=self.tsz, random_state=self.random_state)
        X_train, X_test, y_train, y_test = [df.reset_index(drop=True) for df in (X_train, X_test, y_train, y_test)]
        
        # make train dataset and dataloader
        train_dataset = FinData(X_train, y_train, self.tokenizer, self.maxlen)
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # make test dataset and dataloader
        test_dataset = FinData(X_test, y_test, self.tokenizer, self.maxlen)
        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size*2, shuffle=True)
        
        return train_dl, test_dl
    
if __name__ == '__main__':
    pass