import os, sys
import hashlib
import shutil
import time

def calculate_checksum(file_path, algorithm='sha256'):
    """
    Calculate the checksum of a file using the specified algorithm.
    """
    hash_func = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def copy_and_verify(src, dest, algorithm='sha256', max_retries=3, delay=2):
    """
    Copy a file from src to dest and verify the checksum.
    """
    for attempt in range(max_retries):
        try:
            # Copy the file
            shutil.copy2(src, dest)
            
            # Calculate the checksums
            src_checksum = calculate_checksum(src, algorithm)
            dest_checksum = calculate_checksum(dest, algorithm)
            
            # Verify the checksums
            if src_checksum == dest_checksum:
                print(f"File copied successfully and checksums match for {os.path.basename(src)}")
                return True
            else:
                print(f"Checksum mismatch for {os.path.basename(src)} (Attempt {attempt + 1}/{max_retries})")
        
        except Exception as e:
            print(f"Error copying {src} to {dest}: {e} (Attempt {attempt + 1}/{max_retries})")
        
        # Wait before retrying
        time.sleep(delay)
    
    # All attempts failed
    return False

def copy_files_with_verification(src_dir, dest_dir, algorithm='sha256', max_retries=3, delay=2):
    """
    Copy all files from src_dir to dest_dir and verify their checksums.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    head, exp_name = os.path.split(src_dir)
    print('exp_name, ',exp_name)

    if not os.path.exists(os.path.join(dest_dir, exp_name)):
        os.makedirs(os.path.join(dest_dir, exp_name))

    files_copied = 0
    files_failed = 0

    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir,f)) and '.nd2' in f]
    if len(files)!=1:
        print('error, should have only 1 un-split file, EXIT')
        sys.exit(3)
    else:
        if copy_and_verify(os.path.join(src_dir,files[0]), os.path.join(dest_dir, exp_name, files[0]), algorithm, max_retries, delay):
            files_copied += 1
        else:
            files_failed += 1

    dirs = [f for f in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir,f))]

    for dir in dirs:
        print('dir = ',dir)
        well_dir = os.path.join(dest_dir, '{}_{}'.format(exp_name, dir))
        if not os.path.exists(well_dir):
            for sub_dir in ['raw_files', 'other_files']:
                os.makedirs(os.path.join(well_dir, sub_dir))
        files = [f for f in os.listdir(os.path.join(src_dir, dir)) if os.path.isfile(os.path.join(src_dir, dir, f)) and '.nd2' in f]
        for f in files:

            src_file_path  = os.path.join(src_dir, dir, f)
            dest_file_path = os.path.join(dest_dir, well_dir, 'raw_files', f)
            print('src_file_path  ',src_file_path)
            print('dest_file_path ',dest_file_path)
            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)

            # Copy the file and verify the checksum
            if copy_and_verify(src_file_path, dest_file_path, algorithm, max_retries, delay):
                files_copied += 1
            else:
                files_failed += 1

    print(f"Total files copied: {files_copied}")
    print(f"Total files failed: {files_failed}")

# Example usage
#src_directory  = '/Volumes/upoates/common/Clement/TEST/exp001'
#dest_directory = '/Volumes/upoates/common/Clement/TEST/exp_copy'
#copy_files_with_verification(src_directory, dest_directory)
