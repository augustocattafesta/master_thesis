import xraydb

ka = [xraydb.xray_line('Mn', f'Ka{i}') for i in range(1, 4)]
kb = [xraydb.xray_line('Mn', f'Kb{i}') for i in range(1, 6, 2)]

def weigthed_position(lines):
    pos = sum(line.energy * line.intensity for line in lines) / sum (line.intensity for line in lines)

    return pos

print(weigthed_position(ka))
print(weigthed_position(kb))
