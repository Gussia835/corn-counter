import os
import sys
import argparse
import subprocess


def run_script(script_path, description):
    if not os.path.exists(script_path):
        print(f"Error: Script '{script_path}' not found.")
        return False

    print(f'Running: {description}')
    result = subprocess.run([sys.executable, script_path], 
                            cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        print(f'Execution failed: {description}')
        return False

    print(f'Completed: {description}\n')
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Corn Counter Pipeline Manager')

    parser.add_argument('--prepare',
                        action='store_true',
                        help='Run file_manager.py')

    parser.add_argument('--train-yolo',
                        action='store_true',
                        help='Run Yolo/yolo_model.py')

    parser.add_argument('--train-csrnet',
                        action='store_true',
                        help='Run CSRNet/csrnet.py')

    parser.add_argument('--evaluate',
                        action='store_true',
                        help='Run tester.py')

    parser.add_argument('--visualize',
                        type=str,
                        help='Run visualize.py on a specific image')

    args = parser.parse_args()

    if not any([args.prepare, args.train_yolo, args.train_csrnet, args.evaluate, args.visualize]):
        parser.print_help()
        sys.exit(0)

    if args.prepare:
        run_script('file_manager.py', 'Data preparation')

    if args.train_yolo:
        run_script(os.path.join('Yolo', 'yolo_model.py'), 'YOLO training')

    if args.train_csrnet:
        run_script(os.path.join('CSRNet', 'csrnet.py'), 'CSRNet training')

    if args.evaluate:
        run_script('tester.py', 'Model evaluation')

    if args.visualize:
        cmd = [sys.executable, 'visualize_work.py', '--img', args.visualize]
        subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    print('Pipeline execution finished')
