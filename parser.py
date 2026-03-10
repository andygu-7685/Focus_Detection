import re
import pandas as pd

def parse_log_to_excel(file_path, output_excel):
    with open(file_path, 'r') as f:
        content = f.read()

    # Regex to find each "Current Folder" block and the subsequent rankings
    folder_blocks = re.split(r'Current Folder: ', content)
    
    data_rows = []

    for block in folder_blocks:
        if not block.strip():
            continue
            
        # Extract the Stack ID (e.g., AF_Stacks_20251006_200301)
        stack_match = re.search(r'(AF_Stacks_\d+_\d+)', block)
        # Extract the Quarter (e.g., Q1, Q2...)
        q_match = re.search(r'\\(Q[1-4])', block)
        
        if stack_match and q_match:
            stack_id = stack_match.group(1)
            quarter = q_match.group(1)
            
            # Find all ranked images and their scores in this block
            # Example: diag_X_V59.210.png : 194.40%
            rankings = re.findall(r'diag_(X[+-]?1?)_V([\d\.]+)\.png\s*:\s*([\d\.]+)%', block)
            
            # Temporary storage for this row
            row_entry = {"Stack": stack_id, "Quarter": quarter}
            
            for rel_pos, voltage, score in rankings:
                # Map X-1, X, X+1 to the correct columns
                col_prefix = rel_pos.lower() # x-1, x, or x+1
                row_entry[f"{col_prefix}_V"] = voltage
                # Note: If you want to include the scores, you can add them here too
            
            data_rows.append(row_entry)

    # Convert to DataFrame
    df = pd.DataFrame(data_rows)

    # Pivot/Format to match the Excel Image
    # We want Stack and Quarter as the index, and x-1, x, x+1 as columns
    columns_order = ['x-1_V', 'x_V', 'x+1_V']
    final_df = df.set_index(['Stack', 'Quarter'])[columns_order].unstack(level=-1).stack(level=0)
    
    # Save to Excel
    final_df.to_excel(output_excel)
    print(f"Extraction complete. Data saved to {output_excel}")

# Usage
parse_log_to_excel('log_file.txt', 'Parsed_Results.xlsx')