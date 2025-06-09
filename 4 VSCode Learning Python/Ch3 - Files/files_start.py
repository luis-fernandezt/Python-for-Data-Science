# Read and write files using the built-in Python file methods

#help(open)

def main():  
    # Open a file for writing and create it if it doesn't exist
    #file = open("file.txt", mode="w+")
        
    # Open the file for appending text to the end
    #file = open("file.txt","a+")

    # write some lines of data to the file
    #for i in range(10):
     # file.write("This is line with w+\n")

    #for i in range(10):
     # file.write("This is new line with a+\n")
    
    # close the file when done
    #file.close()
    
    # Open the file and read the contents
    file = open("file.txt", "r")
    if file.mode == 'r':
        contents = file.read()
        print(contents)
    


        #file_backup = file.readlines()
        #for x in file_backup:
         #   print(x)

if __name__ == "__main__":
    main()