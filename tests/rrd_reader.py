"""Recording inspection across 0.26's dataframe API and newer raw chunk API."""
try:
    from rerun.chunk import RrdReader
except ImportError:
    from collections import defaultdict
    from dataclasses import dataclass

    import pyarrow as pa
    import rerun.dataframe as rdf

    @dataclass
    class _Chunk:
        entity_path: str
        batch: object
        is_static: bool
        clear_times: dict

        def to_record_batch(self):
            return self.batch

    class RrdReader:
        def __init__(self, path):
            self.recording = rdf.load_recording(path)

        def blueprints(self):
            # The 0.26 dataframe API exposes data recordings only.
            return ()

        def stream(self, store=None):
            columns = defaultdict(list)
            for column in self.recording.schema().component_columns():
                columns[column.entity_path].append(column)
            for entity, descriptors in columns.items():
                indexed_text = entity.endswith('/events/log') or '/streams/' in entity or '/bookkeeping/' in entity
                index = 'event_sequence' if indexed_text else 'log_tick'
                view = self.recording.view(index=index, contents=entity)
                for static in (True, False):
                    components = {column.name: column.component for column in descriptors if column.is_static == static}
                    if not components:
                        continue
                    table = (view.select_static() if static else view.select()).read_all()
                    keep = [field.name for field in table.schema
                            if field.name in components or (not static and field.name in
                                {column.name for column in self.recording.schema().index_columns()})]
                    data_names = [key for key in keep if key in components]
                    if not table.num_rows or not data_names:
                        continue
                    table = table.select(keep)
                    clears = {}
                    if not static:
                        values = [table.column(key).to_pylist() for key in data_names]
                        cleared = [any(value == [] for value in row) and all(value is None or value == [] for value in row)
                                   for row in zip(*values)]
                        for column in self.recording.schema().index_columns():
                            if column.name in table.schema.names:
                                ticks = table.column(column.name).cast(pa.int64()).to_pylist()
                                clears[column.name] = tuple(tick for tick, clear in zip(ticks, cleared) if clear and tick is not None)
                        # Dataframe expands recursive clears into empty batches
                        # on descendant components. Keep that evidence separately.
                        for key in data_names:
                            array = table.column(key)
                            values = [None if value == [] else value for value in array.to_pylist()]
                            table = table.set_column(table.schema.get_field_index(key), table.schema.field(key),
                                                     pa.array(values, type=array.type))
                    fields = [field.with_name(components.get(field.name, field.name)) for field in table.schema]
                    table = pa.Table.from_arrays(table.columns, schema=pa.schema(fields))
                    if not static:
                        ids = table.column(index).cast(pa.int64())
                        table = table.append_column('rerun.controls.RowId', ids)
                    for batch in table.to_batches():
                        yield _Chunk(entity, batch, static, clears)


def clear_times(chunks, path, timeline='sandbox_elapsed'):
    result = []
    for chunk in chunks:
        if chunk.entity_path == '/' + path:
            batch = chunk.to_record_batch()
            if 'Clear:is_recursive' in batch.schema.names:
                result.extend(tick for tick, value in zip(batch.column(timeline).cast('int64').to_pylist(),
                                                         batch.column('Clear:is_recursive').to_pylist()) if value)
        if chunk.entity_path == '/' + path or chunk.entity_path.startswith('/' + path + '/'):
            result.extend(getattr(chunk, 'clear_times', {}).get(timeline, ()))
    return sorted(set(result))
