from concurrent.futures import ThreadPoolExecutor

from xander_agent import eventlog


def test_event_logs_rotate_and_keep_latest_entries(tmp_path, monkeypatch):
    monkeypatch.setattr('xander_agent.paths.logs_dir', lambda: tmp_path)
    monkeypatch.setattr(eventlog, '_MAX_LOG_BYTES', 1024)
    for index in range(50):
        eventlog.append_global_event('default', tmp_path, 'task-1', 'phase', f'entry {index} café')
    for relative in ('xander.log', 'default.log', 'tasks/task-1.log'):
        target = tmp_path / relative
        backup = target.with_suffix('.log.1')
        assert target.stat().st_size <= 1024
        assert backup.stat().st_size <= 1024
        assert target.stat().st_mode & 0o777 == 0o600
        assert backup.stat().st_mode & 0o777 == 0o600
        assert 'entry 49 café' in target.read_text()
        assert 'entry 0 café' not in backup.read_text()
        assert all(line.startswith('[') for line in backup.read_text().splitlines())
    assert len(list(tmp_path.rglob('*.1'))) == 3
    assert not list(tmp_path.rglob('*.2'))


def test_concurrent_event_writers_preserve_whole_lines(tmp_path, monkeypatch):
    monkeypatch.setattr('xander_agent.paths.logs_dir', lambda: tmp_path)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda i: eventlog.append_global_event(
            'default', tmp_path, 'task-1', 'phase', f'entry {i}'
        ), range(30)))
    for relative in ('xander.log', 'default.log', 'tasks/task-1.log'):
        lines = (tmp_path / relative).read_text().splitlines()
        assert len(lines) == 30
        assert len(set(lines)) == 30
        assert all(line.startswith('[') for line in lines)
