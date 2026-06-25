#!/bin/bash

# List all devices in /dev directory that start with md
devices=$(ls -1 /dev/md* 2>/dev/null)

# Check if there are any matching devices
if [ -z "$devices" ]; then
  echo "No matching devices found."
  exit 1
fi

# Unmount devices and deactivate RAID
for device in $devices; do
  echo "Unmounting device: $device"
  umount "$device" 2>/dev/null
  mdadm --stop "$device" 2>/dev/null
done

echo "All devices have been unmounted and deactivated."

umount /dev/nvme0n1

mdadm --misc --zero-superblock /dev/nvme{0,1,2,3,4,5,6,7,8,9,10,11}n1

index_num=12
start_num=0
end_num=11

# Calculate the number of devices
num_devices=$((end_num - start_num + 1))

# Check if the number of devices is positive
if [ "$num_devices" -lt 1 ]; then
  echo "Invalid device range."
  exit 1
fi

# Create RAID group
echo "Creating RAID group with $num_devices nvme devices"
mdadm --create "/dev/md$index_num" --auto=yes --level=0 -n "$num_devices" $(eval echo /dev/nvme{${start_num}..${end_num}}n1)

# Format RAID device
mkfs.ext4 "/dev/md$index_num"

# Create mount directory
mount_dir="/share/data$num_devices"
mkdir -p "$mount_dir"

# Mount RAID device
mount "/dev/md$index_num" "$mount_dir"

echo "RAID has been created and mounted to $mount_dir"

chmod 777 "$mount_dir"

echo "Permissions for $mount_dir have been changed"
