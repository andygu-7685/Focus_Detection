import re
import pandas as pd

def parse_structured_log(file_path, output_excel):
    # This regex now captures the Score (percentage), Voltage, X, Q, Time, and Folder
    pattern = r"diag_.*?\.png\s*:\s*([\d\.]+)%\s*V=([\d\.]+)\s*x=(-?\d+)\s*Q=(\d+)\s*time=([\d\.]+)ms\s*folder=(AF_Stacks_\d+_\d+)"

    data_rows = []

    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    matches = re.findall(pattern, content)

    for score, v, x, q, time, stack in matches:
        data_rows.append({
            'Stack': stack,
            'Quarter': f"Q{q}",
            'x_val': int(x),
            'Score': score,
            'Voltage': v,
            'Time': time
        })

    df = pd.DataFrame(data_rows)
    if df.empty:
        print("No matching log data found.")
        return

    # Clean duplicates
    df = df.drop_duplicates(subset=['Stack', 'Quarter', 'x_val'], keep='last')

    # Pivot all three metrics
    pivot_v = df.pivot(index=['Stack', 'Quarter'], columns='x_val', values='Voltage')
    pivot_t = df.pivot(index=['Stack', 'Quarter'], columns='x_val', values='Time')
    pivot_s = df.pivot(index=['Stack', 'Quarter'], columns='x_val', values='Score')

    # Build the final table structure
    column_mapping = {-1: 'x-1', 0: 'x', 1: 'x+1'}
    final_df = pd.DataFrame(index=pivot_v.index)
    
    for val in [-1, 0, 1]:
        label = column_mapping[val]
        # Adding Score, Voltage, and Time for each x-position
        final_df[f"{label}_Score%"] = pivot_s[val] if val in pivot_s.columns else ""
        final_df[f"{label}_V"] = pivot_v[val] if val in pivot_v.columns else ""
        final_df[f"{label}_time"] = pivot_t[val] if val in pivot_t.columns else ""

    # Sort columns logically
    ordered_cols = [
        'x-1_Score%', 'x-1_V', 'x-1_time', 
        'x_Score%', 'x_V', 'x_time', 
        'x+1_Score%', 'x+1_V', 'x+1_time'
    ]
    final_df = final_df[ordered_cols]

    # Save to Excel
    try:
        # We use the xlsxwriter engine to get better formatting
        writer = pd.ExcelWriter(output_excel, engine='xlsxwriter')
        final_df.to_excel(writer, sheet_name='Focus Analysis')
        
        # Access the workbook/worksheet for visual formatting
        workbook  = writer.book
        worksheet = writer.sheets['Focus Analysis']
        
        # Add a header format (Bold)
        header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
        for col_num, value in enumerate(final_df.columns.values):
            worksheet.write(0, col_num + 2, value, header_format)

        writer.close()
        print(f"Successfully saved to {output_excel}")
    except Exception as e:
        print(f"Failed to save: {e}. Ensure 'xlsxwriter' is installed (pip install xlsxwriter).")

if __name__ == "__main__":
    parse_structured_log('log_file.txt', 'Focus_Analysis_Complete.xlsx')