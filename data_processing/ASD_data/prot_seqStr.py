
import pandas as pd
import ast
import re

def remove_trailing_X(string, original_list):
    # Define a regular expression pattern to match a series of 'X's at the end of a string
    pattern = r'X+$'

    # Use re.search() to find the index of the start of the matched pattern
    match = re.search(pattern, string)

    if match:
        # Get the start index of the matched pattern
        start_index = match.start()

        # Remove the matched pattern from the string
        modified_string = re.sub(pattern, '', string)

        # Remove corresponding indices from the list
        modified_list = original_list[:start_index]

        return modified_string, modified_list
    else:
        # If no match is found, return the original string and list
        return string, original_list

# Function to convert list of characters to string
def list_to_string(char_list):
    string = ''.join(char_list)
    #string = remove_trailing_X(string)
    return string


#def list_to_string(char_list):
 #   return ''.join(map(str, char_list))

def main():
    root_dir = "/home/lshre1/Documents/PredictingAllostericSites/"
    filename = 'ASD_Release_202306_XF_Allosteric_Contacts_7A_SSP.csv'
    filepath = root_dir+filename

    df = pd.read_csv(filepath)
    print(df['Sequences'].head())
    print(df['Sequences'].apply(type).value_counts())


    df['Sequences'] = df['Sequences'].apply(ast.literal_eval)
    df['Labels'] = df['Labels'].apply(ast.literal_eval)

    #Sequences = []
    # Apply the function to each element in the column
    df['Sequences'] = df['Sequences'].apply(list_to_string)
    

   # df['sequence'] = df['sequence'].apply(lambda x: ''.join(x))
    df[['Sequences', 'Labels']] = df.apply(lambda row: remove_trailing_X(row['Sequences'], row['label']), axis=1, result_type='expand')

    print(df.head())

    #df.drop(columns=['sequence', 'label'], inplace=True)
    print(df.head())
    df.to_csv('ASD_Release_202306_XF_Allosteric_Contacts_7A_SSP.csv', index=False)





if __name__ == "__main__":
    main()