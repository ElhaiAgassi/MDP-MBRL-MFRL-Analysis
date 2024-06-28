def parse_tests():
    tests = []
    with open('GridWorld.py', 'r') as file:
        current_test = {}
        for line in file:
            line = line.strip()
            if line.startswith('#t'):
                if current_test:
                    tests.append(current_test)
                current_test = {}
            elif '=' in line:
                key, value = line.split('=')
                key = key.strip()
                value = value.strip()
                if key in ['w', 'h']:
                    current_test[key] = int(value)
                elif key == 'L':
                    current_test[key] = eval(value)
                elif key in ['p', 'r']:
                    current_test[key] = float(value)
    if current_test:
        tests.append(current_test)
    return tests

