import json
import argparse
import sys
import os

def convert_notebook_to_md(notebook_path, output_path):
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook file '{notebook_path}' not found.")
        sys.exit(1)

    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error reading notebook: {e}")
        sys.exit(1)

    with open(output_path, 'w', encoding='utf-8') as out:
        title = os.path.basename(notebook_path)
        out.write(f'# {title} - Full Export\n\n')
        
        image_counter = 0
        for cell in nb.get('cells', []):
            if cell['cell_type'] == 'markdown':
                # Write Markdown cell content
                out.write(''.join(cell.get('source', [])) + '\n\n')
            
            elif cell['cell_type'] == 'code':
                # Write the Python code
                source_code = ''.join(cell.get('source', []))
                exec_count = cell.get("execution_count", " ")
                if exec_count is None:
                    exec_count = " "
                
                out.write(f'**In [{exec_count}]:**\n')
                out.write('```python\n')
                out.write(source_code)
                if source_code and not source_code.endswith('\n'):
                    out.write('\n')
                out.write('```\n\n')
                
                # Write the outputs
                outputs = cell.get('outputs', [])
                if outputs:
                    text_started = False
                    
                    for output in outputs:
                        if output['output_type'] == 'stream':
                            if not text_started:
                                out.write('**Output:**\n```text\n')
                                text_started = True
                            out.write(''.join(output['text']))
                            
                        elif output['output_type'] in ('display_data', 'execute_result'):
                            if 'image/png' in output.get('data', {}):
                                # If we were writing text, close the code block temporarily
                                if text_started:
                                    out.write('```\n\n')
                                    text_started = False
                                
                                img_data = output['data']['image/png']
                                if isinstance(img_data, list):
                                    img_data = ''.join(img_data)
                                
                                # Clean up base64 string
                                img_data = img_data.replace('\n', '').replace('\r', '').strip()
                                
                                # Embed image directly using base64 data URI
                                out.write(f'![Output Image](data:image/png;base64,{img_data})\n\n')
                                image_counter += 1
                                
                            elif 'text/plain' in output.get('data', {}):
                                if not text_started:
                                    out.write('**Output:**\n```text\n')
                                    text_started = True
                                out.write(''.join(output['data']['text/plain']) + '\n')
                                
                        elif output['output_type'] == 'error':
                            if not text_started:
                                out.write('**Output:**\n```text\n')
                                text_started = True
                            out.write(f"{output.get('ename', 'Error')}: {output.get('evalue', '')}\n")
                            
                    if text_started:
                        out.write('```\n\n')

    print(f"Successfully exported notebook to '{output_path}'")
    if image_counter > 0:
        print(f"Embedded {image_counter} images directly into the markdown file.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a Jupyter Notebook to Markdown (includes Markdown cells, Python code, and Outputs with Inline Base64 Images).')
    parser.add_argument('-i', '--input', default='Bangalore_House_Price_Prediction_V2.ipynb', help='Path to the input .ipynb file')
    parser.add_argument('-o', '--output', default='Bangalore_House_Price_Prediction_V2_Full.md', help='Path to the output .md file')
    
    args = parser.parse_args()
    convert_notebook_to_md(args.input, args.output)
