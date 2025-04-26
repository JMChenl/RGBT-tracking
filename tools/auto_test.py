import subprocess

def test_checkpoints():
    for x in range(17, 26):
        checkpoint_file = f'Dul_SiamCAR_6/checkpoint_e{x}.pth'
        save_name = f'Dul_6/ST_013/test_{x}'
        command = f'python test.py --snapshot {checkpoint_file} --model_name {save_name}'
        subprocess.run(command, shell=True)

def main():
    test_checkpoints()


if __name__ == '__main__':
    main()