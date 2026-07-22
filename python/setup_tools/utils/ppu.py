from pathlib import Path

def install_extension(*args, **kargs):
    # Prevent nvidia driver from being activated when using ppu
    drvfile = Path(__file__).parent.parent.parent.parent / 'third_party' / 'nvidia' / 'backend' / 'driver.py'
    if drvfile.exists():
        with open(drvfile, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if 'def is_active():' in line:
                if i + 1 < len(lines) and 'return False' not in lines[i + 1]:
                    lines.insert(i + 1, '        return False\n')
                break
        with open(drvfile, 'w') as f:
            f.writelines(lines)