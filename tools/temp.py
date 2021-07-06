with open('video.txt') as f:
    contents = f.readlines()

video_results = []
for line in contents:
    line = line.strip()
    line = line.split('|')[-1]
    video_results.append(float(line))

with open('fusion.txt') as f:
    contents = f.readlines()

fusion_results = []
for line in contents:
    line = line.strip()
    line = line.split('|')[-1]
    fusion_results.append(float(line))

video_better_idx = []
for i in range(len(video_results)):
    if video_results[i] > fusion_results[i]:
        video_better_idx.append(i)

print(video_better_idx)