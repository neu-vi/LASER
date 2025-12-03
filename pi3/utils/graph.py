import uuid


class Vertex:
    def __init__(self, data=None, vid=None, default_cache=None):
        self._data = data
        self._vid = vid
        self._uuid = uuid.uuid4()
        self._connectivity = []
        self._edge_weights = []
        self.cache = default_cache

    def __repr__(self):
        sep = '---------------------------------------------\n'
        out_str = sep + f'Vertex <{self._vid if self._vid is not None else self._uuid}>'
        if len(self._connectivity) == 0:
            return out_str

        out_str += ' connected to:\n'
        for idx, v in enumerate(self._connectivity):
            out_str += f'   <{v.vid if v.vid is not None else v.uuid}> - {self._edge_weights[idx]}\n'
        return out_str

    def __eq__(self, other):
        if isinstance(other, Vertex):
            return self.uuid == other.uuid
        return False

    @property
    def uuid(self):
        return self._uuid

    @property
    def vid(self):
        return self._vid

    @property
    def connectivity(self):
        return self._connectivity

    @property
    def edge_weights(self):
        return self._edge_weights

    @property
    def data(self):
        return self._data

    def add_edge(self, vertex, edge_weight):
        self._connectivity.append(vertex)
        self._edge_weights.append(edge_weight)

    def add_edges(self, vertices, edge_weights):
        self._connectivity.extend(vertices)
        self._edge_weights.extend(edge_weights)

    def remove_edge_idx(self, idx: int):
        del self._connectivity[idx]
        del self._edge_weights[idx]

    def remove_edge_indices(self, indices):
        for idx in sorted(set(indices), reverse=True):
            self.remove_edge_idx(idx)

    def remove_edge_vertex(self, vertex: 'Vertex', edge_weights=None):
        if edge_weights is None:
            idx = self._connectivity.index(vertex)
            self.remove_edge_idx(idx)
            return

        all_indices = [i for i, x in enumerate(self._connectivity) if x == vertex]
        for idx in all_indices:
            if self._edge_weights[idx] == edge_weights:
                self.remove_edge_idx(idx)
                return

        raise KeyError('Cannot remove vertex with matching condition')

    def remove_all_edges(self):
        self._connectivity = []
        self._edge_weights = []

    def cut_edge_threshold(self, thresh):
        remove_indices = []
        for idx in range(len(self._connectivity)):
            if self._edge_weights[idx] < thresh:
                remove_indices.append(idx)
            self._connectivity[idx].cut_threshold(thresh)

        self.remove_edge_indices(remove_indices)

    def data_cache_op(self, op_func):
        return op_func(self._data, self.cache)

    def propagate_data_once(self, prop_func):
        for v, edge_wt in zip(self._connectivity, self.edge_weights):
            prop_func(self, v, edge_wt)

    def propagate_data_all(self, prop_func):
        for v, edge_wt in zip(self._connectivity, self.edge_weights):
            prop_func(self, v, edge_wt)
            v.propagate_cache(prop_func)
