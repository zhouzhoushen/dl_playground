import torch
import os
import pandas as pd

def main():
    print(torch.cuda.is_available())

    # Begin.
    os.makedirs(os.path.join("data"), exist_ok = True)
    data_file = os.path.join("data", "house_tiny.csv")
    with open(data_file, "w") as f:
        f.write("NumRooms,Alley,Price\n")
        f.write("NA,Pave,127500\n")
        f.write("2,NA,106000\n")
        f.write("4,NA,178100\n")
        f.write("NA,NA,140000\n")
    
    data = pd.read_csv(data_file)
    print(data)
    
    inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
    inputs = inputs.fillna(inputs.mean(numeric_only = True))
    print(inputs)
    
    inputs = pd.get_dummies(inputs, dummy_na = True)
    print(inputs)
    
    x, y = torch.tensor(inputs.values.astype(float)), torch.tensor(outputs.values.astype(float))
    print("%s\n%s" % (x, y))
    # End.

if __name__ == "__main__":
    main()