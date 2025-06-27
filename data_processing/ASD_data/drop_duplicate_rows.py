import pandas as pd

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'ASD_Release_202306_XF_Allosteric_Contacts.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)
print(len(df))
df = df[df['Sequences'].notna() & (df['Sequences'] != "")]

# Drop duplicate rows
df_unique = df.drop_duplicates()


# Optionally, save the DataFrame with unique rows to a new CSV file
df_unique.to_csv('ASD_Release_202306_XF_Allosteric_Contacts.csv', index=False)

# Display the DataFrame with unique rows
print(df_unique)
print(len(df_unique))
