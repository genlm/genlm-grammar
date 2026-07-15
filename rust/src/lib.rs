use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::cmp::Reverse;

// ── Symbol: unified u32 encoding ──────────────────────────────────────────────
// After renumber(), nonterminals are small u32s. We assign terminals IDs starting
// above max_nt. A flag array distinguishes them. This keeps the inner loop
// all-integer with no string comparisons.

const SENTINEL: u32 = u32::MAX;

// ── Semiring abstraction ──────────────────────────────────────────────────────
// The Earley engine touches weights only through ⊕ (add), ⊗ (mul), 0, and 1,
// so it is generic over any semiring. PyO3 classes cannot be generic, so each
// instantiation gets a thin concrete shell below.

trait Semiring: Copy + PartialEq {
    const ZERO: Self;
    const ONE: Self;
    fn add(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
}

impl Semiring for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    #[inline(always)]
    fn add(self, other: f64) -> f64 {
        self + other
    }
    #[inline(always)]
    fn mul(self, other: f64) -> f64 {
        self * other
    }
}

impl Semiring for bool {
    const ZERO: bool = false;
    const ONE: bool = true;
    #[inline(always)]
    fn add(self, other: bool) -> bool {
        self | other
    }
    #[inline(always)]
    fn mul(self, other: bool) -> bool {
        self & other
    }
}

// ── Column ────────────────────────────────────────────────────────────────────
// An Earley chart column at position k.
//
// Items are:
//   Complete:   (I, X)       stored in c_chart
//   Incomplete: (I, X, Ys)   stored in i_chart
//
// `waiting_for` maps a symbol Y to the list of incomplete items whose dot is
// immediately before Y.

#[derive(Clone)]
struct Column<W> {
    k: u32,
    c_chart: HashMap<(u32, u32), W>,
    i_chart: HashMap<(u32, u32, u32), W>,
    waiting_for: HashMap<u32, Vec<(u32, u32, u32)>>,
}

impl<W> Column<W> {
    fn new(k: u32) -> Self {
        Column {
            k,
            c_chart: HashMap::new(),
            i_chart: HashMap::new(),
            waiting_for: HashMap::new(),
        }
    }
}

// ── Priority queue item for ATTACH ────────────────────────────────────────────
// We use Reverse so BinaryHeap (max-heap) acts as a min-heap on priority.
// Priority = (span_length * ORDER_MAX + order[X]), processed smallest first
// (i.e. shortest spans and highest-order nonterminals first).
#[derive(Eq, PartialEq)]
struct QItem {
    priority: u64,
    item: (u32, u32), // (J, Y) — a complete item
}

impl Ord for QItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // We want to pop the smallest priority first → use Reverse
        other.priority.cmp(&self.priority)
    }
}

impl PartialOrd for QItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ── Core Earley Engine (generic over the semiring) ────────────────────────────

struct EarleyCore<W: Semiring> {
    start: u32,
    order_max: u32,

    /// Nonterminal X → Vec<(weight, Ys_code)>
    rhs: HashMap<u32, Vec<(W, u32)>>,

    /// Nonterminal → topological order bucket
    order: HashMap<u32, u32>,

    /// Left-corner reachability: nonterminal → reachable nonterminals
    outgoing: HashMap<u32, Vec<u32>>,

    /// Ys_code → first symbol (as unified u32 ID)
    first_ys: Vec<u32>,

    /// Ys_code → rest_ys code (tail of dotted rule)
    rest_ys: Vec<u32>,

    /// Ys_code → true if single-symbol suffix
    unit_ys: Vec<bool>,

    /// Ys_code → true if first symbol is a terminal
    is_terminal_flag: Vec<bool>,

    /// Terminal string → unified u32 ID
    terminal_to_id: HashMap<String, u32>,

    /// Unified u32 ID → terminal string (for returning results)
    id_to_terminal: HashMap<u32, String>,

    /// All nonterminal IDs (for is_terminal checks during next_token_weights)
    nonterminals: HashSet<u32>,

    /// Column storage — all columns live here, referenced by index.
    /// A slot is None once reclaimed; its index is recycled via `free_list`.
    columns: Vec<Option<Column<W>>>,

    /// Per-column count of cache entries referencing it
    refcounts: Vec<u32>,

    /// Per-column generation, bumped when a slot is reclaimed; used to
    /// detect expired chart handles instead of silently aliasing them
    generations: Vec<u64>,

    /// Recycled column slots
    free_list: Vec<usize>,

    /// Chart cache: token prefix → (column indices, last-used stamp)
    chart_cache: HashMap<Vec<String>, (Vec<usize>, u64)>,

    /// Eviction order: last-used stamp → cache key (stamps are unique),
    /// giving O(log n) promotion and eviction
    lru_order: BTreeMap<u64, Vec<String>>,

    /// Monotone counter for LRU stamps
    access_counter: u64,

    /// The initial column index (after PREDICT on empty input)
    initial_col_idx: usize,

    /// Weight of empty string from start symbol
    empty_weight: W,

    /// Maximum number of cached prefixes (None = unbounded)
    max_cache_size: Option<usize>,
}

impl<W: Semiring> EarleyCore<W> {
    fn new(
        rhs: HashMap<u32, Vec<(W, u32)>>,
        start: u32,
        order: HashMap<u32, u32>,
        order_max: u32,
        outgoing: HashMap<u32, Vec<u32>>,
        first_ys: Vec<u32>,
        is_terminal_flags: Vec<bool>,
        rest_ys: Vec<u32>,
        unit_ys: Vec<bool>,
        terminal_to_id: HashMap<String, u32>,
        id_to_terminal: HashMap<u32, String>,
        nonterminals: HashSet<u32>,
        empty_weight: W,
        max_cache_size: Option<usize>,
    ) -> Self {
        let mut earley = EarleyCore {
            start,
            order_max,
            rhs,
            order,
            outgoing,
            first_ys,
            rest_ys,
            unit_ys,
            is_terminal_flag: is_terminal_flags,
            terminal_to_id,
            id_to_terminal,
            nonterminals,
            columns: Vec::new(),
            refcounts: Vec::new(),
            generations: Vec::new(),
            free_list: Vec::new(),
            chart_cache: HashMap::new(),
            lru_order: BTreeMap::new(),
            access_counter: 0,
            initial_col_idx: 0,
            empty_weight,
            max_cache_size,
        };

        // Build initial column (k=0) with PREDICT
        let mut col = Column::new(0);
        earley.predict(&mut col);
        earley.initial_col_idx = earley.columns.len();
        earley.columns.push(Some(col));
        earley.refcounts.push(1); // pinned: never reclaimed
        earley.generations.push(0);

        earley
    }

    /// Parse a token sequence, return its weight (prefix probability).
    fn parse(&mut self, tokens: Vec<String>) -> W {
        let n = tokens.len();
        if n == 0 {
            return self.empty_weight;
        }

        let col_indices = self.chart_inner(&tokens);
        let last_idx = col_indices[n];
        let col = self.col(last_idx);

        col.c_chart.get(&(0, self.start)).copied().unwrap_or(W::ZERO)
    }

    /// Compute chart for a token sequence. Returns an opaque chart ID
    /// (index into internal cache).
    fn chart(&mut self, tokens: Vec<String>) -> Vec<usize> {
        self.chart_inner(&tokens)
    }

    /// Compute next-token weights given a chart (list of column indices).
    /// Returns a dict: terminal_string → weight.
    fn next_token_weights(&self, col_indices: Vec<usize>) -> HashMap<String, W> {
        self.next_token_weights_inner(&col_indices)
    }

    /// Clear the chart cache.
    fn clear_cache(&mut self) {
        self.chart_cache.clear();
        self.lru_order.clear();
        // Keep only the initial column (its generation is preserved)
        let initial = self.columns[self.initial_col_idx].take();
        let gen = self.generations[self.initial_col_idx];
        self.columns.clear();
        self.refcounts.clear();
        self.generations.clear();
        self.free_list.clear();
        self.initial_col_idx = 0;
        self.columns.push(initial);
        self.refcounts.push(1);
        self.generations.push(gen);
    }

    /// Number of cached prefixes (diagnostics).
    fn cache_len(&self) -> usize {
        self.chart_cache.len()
    }

    /// Number of live (non-reclaimed) columns (diagnostics).
    fn live_columns(&self) -> usize {
        self.columns.iter().filter(|c| c.is_some()).count()
    }
}

// ── Internal implementation ───────────────────────────────────────────────────

impl<W: Semiring> EarleyCore<W> {
    fn col(&self, i: usize) -> &Column<W> {
        self.columns[i].as_ref().expect("column was reclaimed")
    }

    /// Tag arena indices with their slot generations → an expiry-checkable handle.
    fn tag_handle(&self, indices: &[usize]) -> Vec<(usize, u64)> {
        indices.iter().map(|&i| (i, self.generations[i])).collect()
    }

    /// Check a handle from `tag_handle`; errors if any column was reclaimed.
    fn validate_handle(&self, handle: &[(usize, u64)]) -> PyResult<Vec<usize>> {
        handle
            .iter()
            .map(|&(i, g)| {
                if i < self.columns.len()
                    && self.generations[i] == g
                    && self.columns[i].is_some()
                {
                    Ok(i)
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "expired chart handle: the cached chart was evicted; recompute it with chart()",
                    ))
                }
            })
            .collect()
    }

    fn alloc_column(&mut self, col: Column<W>) -> usize {
        match self.free_list.pop() {
            Some(i) => {
                self.columns[i] = Some(col);
                i
            }
            None => {
                self.columns.push(Some(col));
                self.refcounts.push(0);
                self.generations.push(0);
                self.columns.len() - 1
            }
        }
    }

    /// Insert a chart into the cache; beyond `max_cache_size`, evict the
    /// least-recently-used entry and reclaim columns no cached chart references.
    fn cache_insert(&mut self, key: Vec<String>, indices: &[usize]) {
        for &i in indices {
            self.refcounts[i] += 1;
        }
        self.access_counter += 1;
        self.lru_order.insert(self.access_counter, key.clone());
        self.chart_cache.insert(key, (indices.to_vec(), self.access_counter));
        if let Some(max) = self.max_cache_size {
            while self.chart_cache.len() > max {
                let (_, lru_key) = self.lru_order.pop_first().unwrap();
                let (evicted, _) = self.chart_cache.remove(&lru_key).unwrap();
                for i in evicted {
                    self.refcounts[i] -= 1;
                    // The initial column is pinned by its extra refcount.
                    if self.refcounts[i] == 0 {
                        self.columns[i] = None;
                        self.generations[i] += 1;
                        self.free_list.push(i);
                    }
                }
            }
        }
    }

    fn chart_inner(&mut self, tokens: &[String]) -> Vec<usize> {
        let key = tokens.to_vec();
        self.access_counter += 1;
        let stamp = self.access_counter;
        if let Some(entry) = self.chart_cache.get_mut(&key) {
            let moved = self.lru_order.remove(&entry.1).unwrap();
            self.lru_order.insert(stamp, moved);
            entry.1 = stamp; // mark most recently used
            return entry.0.clone();
        }

        if tokens.is_empty() {
            let result = vec![self.initial_col_idx];
            self.cache_insert(key, &result);
            return result;
        }

        // Recursive: get chart for prefix, then extend
        let prev_indices = self.chart_inner(&tokens[..tokens.len() - 1]);
        let last_token = &tokens[tokens.len() - 1];
        let new_col_idx = self.next_column_inner(&prev_indices, last_token);

        let mut result = prev_indices;
        result.push(new_col_idx);
        self.cache_insert(key, &result);
        result
    }

    fn next_column_inner(&mut self, prev_col_indices: &[usize], token: &str) -> usize {
        let k = self.col(*prev_col_indices.last().unwrap()).k + 1;
        let mut next_col = Column::new(k);

        let token_id = match self.terminal_to_id.get(token) {
            Some(&id) => id,
            None => { // Unknown token → empty column
                self.predict(&mut next_col);
                return self.alloc_column(next_col);
            }
        };

        // We need to collect items from prev_col before mutating next_col.
        // Clone the relevant waiting_for list.
        let prev_col_idx = *prev_col_indices.last().unwrap();

        let scan_items: Vec<(u32, u32, u32)> = self.col(prev_col_idx)
            .waiting_for
            .get(&token_id)
            .cloned()
            .unwrap_or_default();

        let scan_weights: Vec<_> = scan_items
            .iter()
            .map(|item| self.col(prev_col_idx).i_chart[item])
            .collect();

        // SCAN: phrase(I, X/Ys, K) += phrase(I, X/[token|Ys], J) * word(J, token, K)
        let mut q_set: HashSet<(u32, u32)> = HashSet::new();
        let mut queue: BinaryHeap<QItem> = BinaryHeap::new();

        for (item, &weight) in scan_items.iter().zip(scan_weights.iter()) {
            let (i, x, ys) = *item;
            let rest = self.rest_ys[ys as usize];
            Self::update_static(
                &mut next_col, &mut queue, &mut q_set,
                i, x, rest, weight,
                self.order_max, &self.order, &self.first_ys,
                &self.is_terminal_flag,
            );
        }

        // ATTACH: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * phrase(J, Y/[], K)
        while let Some(q_item) = queue.pop() {
            let (j, y) = q_item.item;
            let y_weight = next_col.c_chart[&(j, y)];

            let col_j_idx = prev_col_indices[j as usize];

            let customers: Vec<(u32, u32, u32)> = self.col(col_j_idx)
                .waiting_for
                .get(&y)
                .cloned()
                .unwrap_or_default();

            let customer_weights: Vec<_> = customers
                .iter()
                .map(|item| self.col(col_j_idx).i_chart[item])
                .collect();

            for (customer, &cw) in customers.iter().zip(customer_weights.iter()) {
                let (i, x, ys) = *customer;
                let rest = self.rest_ys[ys as usize];
                Self::update_static(
                    &mut next_col, &mut queue, &mut q_set,
                    i, x, rest, cw.mul(y_weight),
                    self.order_max, &self.order, &self.first_ys,
                    &self.is_terminal_flag,
                );
            }
        }

        // PREDICT
        self.predict(&mut next_col);

        self.alloc_column(next_col)
    }

    /// The innermost update function. Called extremely frequently.
    #[inline(always)]
    fn update_static(
        col: &mut Column<W>,
        queue: &mut BinaryHeap<QItem>,
        q_set: &mut HashSet<(u32, u32)>,
        i: u32, x: u32, ys: u32, value: W,
        order_max: u32,
        order: &HashMap<u32, u32>,
        first_ys: &[u32],
        is_terminal_flag: &[bool],
    ) {
        let k = col.k;
        if ys == 0 {
            // Complete item: phrase(I, X/[], K)
            let item = (i, x);
            if let Some(existing) = col.c_chart.get_mut(&item) {
                *existing = (*existing).add(value);
            } else {
                // New complete item → add to priority queue
                let priority = (k - i) as u64 * order_max as u64
                    + *order.get(&x).unwrap_or(&0) as u64;
                if q_set.insert(item) {
                    queue.push(QItem { priority, item });
                }
                col.c_chart.insert(item, value);
            }
        } else {
            // Incomplete item: phrase(I, X/[Y|...], K)
            let item = (i, x, ys);
            if let Some(existing) = col.i_chart.get_mut(&item) {
                *existing = (*existing).add(value);
            } else {
                let first = first_ys[ys as usize];
                col.waiting_for
                    .entry(first)
                    .or_default()
                    .push(item);
                col.i_chart.insert(item, value);
            }
        }
    }

    fn predict(&self, col: &mut Column<W>) {
        // PREDICT: phrase(K, X/Ys, K) += rule(X -> Ys)
        // with left-corner filtering
        let k = col.k;

        let mut agenda: Vec<u32> = if k == 0 {
            vec![self.start]
        } else {
            // All nonterminals that something is waiting for
            col.waiting_for
                .keys()
                .filter(|sym| self.nonterminals.contains(sym))
                .copied()
                .collect()
        };

        let mut reachable: HashSet<u32> = agenda.iter().copied().collect();

        while let Some(x) = agenda.pop() {
            if let Some(targets) = self.outgoing.get(&x) {
                for &y in targets {
                    if reachable.insert(y) {
                        agenda.push(y);
                    }
                }
            }
        }

        for x in &reachable {
            if let Some(rules) = self.rhs.get(x) {
                for &(w, ys) in rules {
                    // No queue needed during PREDICT (Q = None in Python)
                    if ys == 0 {
                        // Nullary rule (shouldn't happen after nullaryremove, but handle it)
                        let item = (k, *x);
                        let e = col.c_chart.entry(item).or_insert(W::ZERO);
                        *e = (*e).add(w);
                    } else {
                        let item = (k, *x, ys);
                        if let Some(existing) = col.i_chart.get_mut(&item) {
                            *existing = (*existing).add(w);
                        } else {
                            let first = self.first_ys[ys as usize];
                            col.waiting_for
                                .entry(first)
                                .or_default()
                                .push(item);
                            col.i_chart.insert(item, w);
                        }
                    }
                }
            }
        }
    }

    fn next_token_weights_inner(&self, col_indices: &[usize]) -> HashMap<String, W> {
        let last_idx = *col_indices.last().unwrap();
        let col = self.col(last_idx);

        // q(0, S) = 1
        let mut q: HashMap<(u32, u32), W> = HashMap::new();
        q.insert((0, self.start), W::ONE);

        let mut result: HashMap<String, W> = HashMap::new();

        // For each terminal Y that something is waiting for in the last column
        for (&y_id, items) in &col.waiting_for {
            // Check if Y is a terminal
            if !self.id_to_terminal.contains_key(&y_id) {
                continue;
            }

            let mut total = W::ZERO;
            for &(i, x, ys) in items {
                if self.unit_ys[ys as usize] {
                    let node = (i, x);
                    let value = self.helper(node, col_indices, &mut q);
                    total = total.add(col.i_chart[&(i, x, ys)].mul(value));
                }
            }

            if total != W::ZERO {
                let terminal_str = &self.id_to_terminal[&y_id];
                result.insert(terminal_str.clone(), total);
            }
        }

        result
    }

    /// Iterative DFS helper for backward pass (corresponds to Python _helper).
    fn helper(
        &self,
        top: (u32, u32),
        col_indices: &[usize],
        q: &mut HashMap<(u32, u32), W>,
    ) -> W {
        if let Some(&v) = q.get(&top) {
            return v;
        }

        // Stack-based iterative DFS
        struct Frame<V> {
            node: (u32, u32),
            edges: Vec<(u32, u32, u32)>,
            cursor: usize,
            value: V,
        }

        let mut stack: Vec<Frame<W>> = vec![Frame {
            node: top,
            edges: Vec::new(),
            cursor: usize::MAX, // sentinel: edges not yet loaded
            value: W::ZERO,
        }];

        while let Some(frame) = stack.last_mut() {
            let (j, y) = frame.node;

            if frame.cursor == usize::MAX {
                // Load edges (first visit)
                if let Some(&cached) = q.get(&frame.node) {
                    // Already computed by a different path
                    let val = cached;
                    stack.pop();
                    // Contribute to parent
                    if let Some(parent) = stack.last_mut() {
                        if parent.cursor < parent.edges.len() {
                            let (pi, px, _) = parent.edges[parent.cursor];
                            let col_j_idx = col_indices[j as usize];
                            let iw = self.col(col_j_idx).i_chart
                                .get(&(pi, px, parent.edges[parent.cursor].2))
                                .copied()
                                .unwrap_or(Semiring::ZERO);
                            parent.value = parent.value.add(iw.mul(val));
                            parent.cursor += 1;
                        }
                    }
                    continue;
                }

                let col_j_idx = col_indices[j as usize];
                let col_j = self.col(col_j_idx);
                let edges: Vec<(u32, u32, u32)> = col_j
                    .waiting_for
                    .get(&y)
                    .map(|items| {
                        items
                            .iter()
                            .filter(|(_, _, ys)| self.unit_ys[*ys as usize])
                            .copied()
                            .collect()
                    })
                    .unwrap_or_default();
                frame.edges = edges;
                frame.cursor = 0;
            }

            if frame.cursor >= frame.edges.len() {
                // All edges processed — finalize this node
                let node = frame.node;
                let value = frame.value;
                q.insert(node, value);
                stack.pop();

                // Contribute to parent
                if let Some(parent) = stack.last_mut() {
                    if parent.cursor < parent.edges.len() {
                        let (pi, px, pys) = parent.edges[parent.cursor];
                        let (pj, _py) = parent.node;
                        let col_pj_idx = col_indices[pj as usize];
                        let iw = self.col(col_pj_idx).i_chart
                            .get(&(pi, px, pys))
                            .copied()
                            .unwrap_or(Semiring::ZERO);
                        parent.value = parent.value.add(iw.mul(value));
                        parent.cursor += 1;
                    }
                }
            } else {
                // Process next edge
                let (ei, ex, _eys) = frame.edges[frame.cursor];
                let neighbor = (ei, ex);

                if let Some(&cached) = q.get(&neighbor) {
                    // Neighbor already computed
                    let (pi, px, pys) = frame.edges[frame.cursor];
                    let col_j_idx = col_indices[j as usize];
                    let iw = self.col(col_j_idx).i_chart
                        .get(&(pi, px, pys))
                        .copied()
                        .unwrap_or(Semiring::ZERO);
                    frame.value = frame.value.add(iw.mul(cached));
                    frame.cursor += 1;
                } else {
                    // Need to compute neighbor first — push it
                    stack.push(Frame {
                        node: neighbor,
                        edges: Vec::new(),
                        cursor: usize::MAX,
                        value: Semiring::ZERO,
                    });
                }
            }
        }

        q[&top]
    }
}

// ── Python-facing shells (PyO3 classes cannot be generic) ────────────────────

macro_rules! earley_shell {
    ($name:ident, $W:ty) => {
        #[pyclass]
        struct $name {
            core: EarleyCore<$W>,
        }

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (rhs, start, order, order_max, outgoing, first_ys, is_terminal_flags, rest_ys, unit_ys, terminal_to_id, id_to_terminal, nonterminals, empty_weight, max_cache_size=None))]
            #[allow(clippy::too_many_arguments)]
            fn new(
                rhs: HashMap<u32, Vec<($W, u32)>>,
                start: u32,
                order: HashMap<u32, u32>,
                order_max: u32,
                outgoing: HashMap<u32, Vec<u32>>,
                first_ys: Vec<u32>,
                is_terminal_flags: Vec<bool>,
                rest_ys: Vec<u32>,
                unit_ys: Vec<bool>,
                terminal_to_id: HashMap<String, u32>,
                id_to_terminal: HashMap<u32, String>,
                nonterminals: HashSet<u32>,
                empty_weight: $W,
                max_cache_size: Option<usize>,
            ) -> Self {
                $name {
                    core: EarleyCore::new(
                        rhs, start, order, order_max, outgoing, first_ys,
                        is_terminal_flags, rest_ys, unit_ys, terminal_to_id,
                        id_to_terminal, nonterminals, empty_weight, max_cache_size,
                    ),
                }
            }

            /// Parse a token sequence, return its weight (prefix weight).
            fn parse(&mut self, tokens: Vec<String>) -> $W {
                self.core.parse(tokens)
            }

            /// Compute chart for a token sequence. Returns an opaque,
            /// generation-tagged handle for `next_token_weights`.
            fn chart(&mut self, tokens: Vec<String>) -> Vec<(usize, u64)> {
                let indices = self.core.chart(tokens);
                self.core.tag_handle(&indices)
            }

            /// Compute next-token weights given a handle from `chart()`.
            /// Returns a dict: terminal_string → weight.
            /// Raises ValueError if the handle has expired (cache eviction).
            fn next_token_weights(&self, handle: Vec<(usize, u64)>) -> PyResult<HashMap<String, $W>> {
                let col_indices = self.core.validate_handle(&handle)?;
                Ok(self.core.next_token_weights(col_indices))
            }

            /// Clear the chart cache.
            fn clear_cache(&mut self) {
                self.core.clear_cache()
            }

            /// Number of cached prefixes (diagnostics).
            fn cache_len(&self) -> usize {
                self.core.cache_len()
            }

            /// Number of live (non-reclaimed) columns (diagnostics).
            fn live_columns(&self) -> usize {
                self.core.live_columns()
            }
        }
    };
}

earley_shell!(RustEarley, f64);
earley_shell!(RustEarleyBool, bool);

// ── Rescaled Column ──────────────────────────────────────────────────────────

#[derive(Clone)]
struct RescaledColumn {
    k: u32,
    c_chart: HashMap<(u32, u32), f64>,
    i_chart: HashMap<(u32, u32, u32), f64>,
    waiting_for: HashMap<u32, Vec<(u32, u32, u32)>>,
    rescale: f64,
}

impl RescaledColumn {
    fn new(k: u32) -> Self {
        RescaledColumn {
            k,
            c_chart: HashMap::new(),
            i_chart: HashMap::new(),
            waiting_for: HashMap::new(),
            rescale: 1.0,
        }
    }
}

// ── Rescaled Earley Engine ───────────────────────────────────────────────────

#[pyclass]
struct RustEarleyRescaled {
    start: u32,
    order_max: u32,
    rhs: HashMap<u32, Vec<(f64, u32)>>,
    order: HashMap<u32, u32>,
    outgoing: HashMap<u32, Vec<u32>>,
    first_ys: Vec<u32>,
    rest_ys: Vec<u32>,
    unit_ys: Vec<bool>,
    is_terminal_flag: Vec<bool>,
    terminal_to_id: HashMap<String, u32>,
    id_to_terminal: HashMap<u32, String>,
    nonterminals: HashSet<u32>,

    columns: Vec<Option<RescaledColumn>>,
    refcounts: Vec<u32>,
    generations: Vec<u64>,
    free_list: Vec<usize>,
    chart_cache: HashMap<Vec<String>, (Vec<usize>, u64)>,
    lru_order: BTreeMap<u64, Vec<String>>,
    access_counter: u64,
    initial_col_idx: usize,
    empty_weight: f64,
    max_cache_size: Option<usize>,
}

#[pymethods]
impl RustEarleyRescaled {
    #[new]
    #[pyo3(signature = (rhs, start, order, order_max, outgoing, first_ys, is_terminal_flags, rest_ys, unit_ys, terminal_to_id, id_to_terminal, nonterminals, empty_weight, max_cache_size=None))]
    fn new(
        rhs: HashMap<u32, Vec<(f64, u32)>>,
        start: u32,
        order: HashMap<u32, u32>,
        order_max: u32,
        outgoing: HashMap<u32, Vec<u32>>,
        first_ys: Vec<u32>,
        is_terminal_flags: Vec<bool>,
        rest_ys: Vec<u32>,
        unit_ys: Vec<bool>,
        terminal_to_id: HashMap<String, u32>,
        id_to_terminal: HashMap<u32, String>,
        nonterminals: HashSet<u32>,
        empty_weight: f64,
        max_cache_size: Option<usize>,
    ) -> Self {
        let mut earley = RustEarleyRescaled {
            start,
            order_max,
            rhs,
            order,
            outgoing,
            first_ys,
            rest_ys,
            unit_ys,
            is_terminal_flag: is_terminal_flags,
            terminal_to_id,
            id_to_terminal,
            nonterminals,
            columns: Vec::new(),
            refcounts: Vec::new(),
            generations: Vec::new(),
            free_list: Vec::new(),
            chart_cache: HashMap::new(),
            lru_order: BTreeMap::new(),
            access_counter: 0,
            initial_col_idx: 0,
            empty_weight,
            max_cache_size,
        };

        let mut col = RescaledColumn::new(0);
        earley.predict(&mut col);
        col.rescale = 1.0;
        earley.initial_col_idx = earley.columns.len();
        earley.columns.push(Some(col));
        earley.refcounts.push(1); // pinned: never reclaimed
        earley.generations.push(0);

        earley
    }

    /// Parse a token sequence, return its weight (unrescaled).
    fn parse(&mut self, tokens: Vec<String>) -> f64 {
        let n = tokens.len();
        if n == 0 {
            return self.empty_weight;
        }

        let col_indices = self.chart_inner(&tokens);
        let last_idx = col_indices[n];
        let col = self.col(last_idx);
        let value = *col.c_chart.get(&(0, self.start)).unwrap_or(&0.0);

        // Divide by product of rescaling coefficients for cols[0..n]
        let rescale_product = self.rescale_product(&col_indices, 0, n);
        if rescale_product == 0.0 { 0.0 } else { value / rescale_product }
    }

    /// Parse a token sequence, return log of its weight.
    fn logp(&mut self, tokens: Vec<String>) -> f64 {
        let n = tokens.len();
        if n == 0 {
            return self.empty_weight.ln();
        }

        let col_indices = self.chart_inner(&tokens);
        let last_idx = col_indices[n];
        let col = self.col(last_idx);
        let value = *col.c_chart.get(&(0, self.start)).unwrap_or(&0.0);

        value.ln() - self.log_rescale(&col_indices, 0, n)
    }

    fn chart(&mut self, tokens: Vec<String>) -> Vec<(usize, u64)> {
        let indices = self.chart_inner(&tokens);
        self.tag_handle(&indices)
    }

    /// Compute next-token weights given a handle from `chart()`.
    /// Returns a dict: terminal_string → weight.
    /// Raises ValueError if the handle has expired (cache eviction).
    /// Note: rescaling cancels out after normalization, so we compute
    /// raw weights and normalize.
    fn next_token_weights(&self, handle: Vec<(usize, u64)>) -> PyResult<HashMap<String, f64>> {
        let col_indices = self.validate_handle(&handle)?;
        let raw = self.next_token_weights_inner(&col_indices);

        // Normalize: sum all weights, divide each by total
        let total: f64 = raw.values().sum();
        if total == 0.0 {
            return Ok(raw);
        }
        Ok(raw.into_iter().map(|(k, v)| (k, v / total)).collect())
    }

    fn clear_cache(&mut self) {
        self.chart_cache.clear();
        self.lru_order.clear();
        let initial = self.columns[self.initial_col_idx].take();
        let gen = self.generations[self.initial_col_idx];
        self.columns.clear();
        self.refcounts.clear();
        self.generations.clear();
        self.free_list.clear();
        self.initial_col_idx = 0;
        self.columns.push(initial);
        self.refcounts.push(1);
        self.generations.push(gen);
    }

    /// Number of cached prefixes (diagnostics).
    fn cache_len(&self) -> usize {
        self.chart_cache.len()
    }

    /// Number of live (non-reclaimed) columns (diagnostics).
    fn live_columns(&self) -> usize {
        self.columns.iter().filter(|c| c.is_some()).count()
    }
}

// ── Rescaled internal implementation ─────────────────────────────────────────

impl RustEarleyRescaled {
    fn col(&self, i: usize) -> &RescaledColumn {
        self.columns[i].as_ref().expect("column was reclaimed")
    }

    /// Tag arena indices with their slot generations → an expiry-checkable handle.
    fn tag_handle(&self, indices: &[usize]) -> Vec<(usize, u64)> {
        indices.iter().map(|&i| (i, self.generations[i])).collect()
    }

    /// Check a handle from `tag_handle`; errors if any column was reclaimed.
    fn validate_handle(&self, handle: &[(usize, u64)]) -> PyResult<Vec<usize>> {
        handle
            .iter()
            .map(|&(i, g)| {
                if i < self.columns.len()
                    && self.generations[i] == g
                    && self.columns[i].is_some()
                {
                    Ok(i)
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "expired chart handle: the cached chart was evicted; recompute it with chart()",
                    ))
                }
            })
            .collect()
    }

    fn alloc_column(&mut self, col: RescaledColumn) -> usize {
        match self.free_list.pop() {
            Some(i) => {
                self.columns[i] = Some(col);
                i
            }
            None => {
                self.columns.push(Some(col));
                self.refcounts.push(0);
                self.generations.push(0);
                self.columns.len() - 1
            }
        }
    }

    /// Insert a chart into the cache; beyond `max_cache_size`, evict the
    /// least-recently-used entry and reclaim columns no cached chart references.
    fn cache_insert(&mut self, key: Vec<String>, indices: &[usize]) {
        for &i in indices {
            self.refcounts[i] += 1;
        }
        self.access_counter += 1;
        self.lru_order.insert(self.access_counter, key.clone());
        self.chart_cache.insert(key, (indices.to_vec(), self.access_counter));
        if let Some(max) = self.max_cache_size {
            while self.chart_cache.len() > max {
                let (_, lru_key) = self.lru_order.pop_first().unwrap();
                let (evicted, _) = self.chart_cache.remove(&lru_key).unwrap();
                for i in evicted {
                    self.refcounts[i] -= 1;
                    // The initial column is pinned by its extra refcount.
                    if self.refcounts[i] == 0 {
                        self.columns[i] = None;
                        self.generations[i] += 1;
                        self.free_list.push(i);
                    }
                }
            }
        }
    }

    fn rescale_product(&self, col_indices: &[usize], from: usize, to: usize) -> f64 {
        let mut product = 1.0f64;
        for &idx in &col_indices[from..to] {
            product *= self.col(idx).rescale;
        }
        product
    }

    fn log_rescale(&self, col_indices: &[usize], from: usize, to: usize) -> f64 {
        let mut total = 0.0f64;
        for &idx in &col_indices[from..to] {
            total += self.col(idx).rescale.ln();
        }
        total
    }

    fn chart_inner(&mut self, tokens: &[String]) -> Vec<usize> {
        let key = tokens.to_vec();
        self.access_counter += 1;
        let stamp = self.access_counter;
        if let Some(entry) = self.chart_cache.get_mut(&key) {
            let moved = self.lru_order.remove(&entry.1).unwrap();
            self.lru_order.insert(stamp, moved);
            entry.1 = stamp; // mark most recently used
            return entry.0.clone();
        }

        if tokens.is_empty() {
            let result = vec![self.initial_col_idx];
            self.cache_insert(key, &result);
            return result;
        }

        let prev_indices = self.chart_inner(&tokens[..tokens.len() - 1]);
        let last_token = &tokens[tokens.len() - 1];
        let new_col_idx = self.next_column_inner(&prev_indices, last_token);

        let mut result = prev_indices;
        result.push(new_col_idx);
        self.cache_insert(key, &result);
        result
    }

    fn next_column_inner(&mut self, prev_col_indices: &[usize], token: &str) -> usize {
        let prev_col_idx = *prev_col_indices.last().unwrap();
        let k = self.col(prev_col_idx).k + 1;
        let mut next_col = RescaledColumn::new(k);

        let token_id = match self.terminal_to_id.get(token) {
            Some(&id) => id,
            None => {
                self.predict(&mut next_col);
                next_col.rescale = 1.0;
                return self.alloc_column(next_col);
            }
        };

        // Get rescale factor from previous column
        let prev_rescale = self.col(prev_col_idx).rescale;

        let scan_items: Vec<(u32, u32, u32)> = self.col(prev_col_idx)
            .waiting_for
            .get(&token_id)
            .cloned()
            .unwrap_or_default();

        let scan_weights: Vec<_> = scan_items
            .iter()
            .map(|item| self.col(prev_col_idx).i_chart[item])
            .collect();

        // SCAN: multiply by prev_col.rescale for numerical stability
        let mut q_set: HashSet<(u32, u32)> = HashSet::new();
        let mut queue: BinaryHeap<QItem> = BinaryHeap::new();

        for (item, &weight) in scan_items.iter().zip(scan_weights.iter()) {
            let (i, x, ys) = *item;
            let rest = self.rest_ys[ys as usize];
            Self::update_static(
                &mut next_col, &mut queue, &mut q_set,
                i, x, rest, weight * prev_rescale,
                self.order_max, &self.order, &self.first_ys,
                &self.is_terminal_flag,
            );
        }

        // ATTACH
        while let Some(q_item) = queue.pop() {
            let (j, y) = q_item.item;
            let y_weight = next_col.c_chart[&(j, y)];

            let col_j_idx = prev_col_indices[j as usize];

            let customers: Vec<(u32, u32, u32)> = self.col(col_j_idx)
                .waiting_for
                .get(&y)
                .cloned()
                .unwrap_or_default();

            let customer_weights: Vec<_> = customers
                .iter()
                .map(|item| self.col(col_j_idx).i_chart[item])
                .collect();

            for (customer, &cw) in customers.iter().zip(customer_weights.iter()) {
                let (i, x, ys) = *customer;
                let rest = self.rest_ys[ys as usize];
                Self::update_static(
                    &mut next_col, &mut queue, &mut q_set,
                    i, x, rest, cw.mul(y_weight),
                    self.order_max, &self.order, &self.first_ys,
                    &self.is_terminal_flag,
                );
            }
        }

        // PREDICT
        self.predict(&mut next_col);

        // Compute rescaling coefficient
        let num = self.col(prev_col_idx).c_chart
            .get(&(0, self.start)).copied().unwrap_or(Semiring::ZERO);
        let den = next_col.c_chart
            .get(&(0, self.start)).copied().unwrap_or(Semiring::ZERO);

        if den == 0.0 || num == 0.0 {
            next_col.rescale = 1.0;
        } else {
            next_col.rescale = num / den * prev_rescale;
        }

        self.alloc_column(next_col)
    }

    #[inline(always)]
    fn update_static(
        col: &mut RescaledColumn,
        queue: &mut BinaryHeap<QItem>,
        q_set: &mut HashSet<(u32, u32)>,
        i: u32, x: u32, ys: u32, value: f64,
        order_max: u32,
        order: &HashMap<u32, u32>,
        first_ys: &[u32],
        _is_terminal_flag: &[bool],
    ) {
        let k = col.k;
        if ys == 0 {
            let item = (i, x);
            if let Some(existing) = col.c_chart.get_mut(&item) {
                *existing = (*existing).add(value);
            } else {
                let priority = (k - i) as u64 * order_max as u64
                    + *order.get(&x).unwrap_or(&0) as u64;
                if q_set.insert(item) {
                    queue.push(QItem { priority, item });
                }
                col.c_chart.insert(item, value);
            }
        } else {
            let item = (i, x, ys);
            if let Some(existing) = col.i_chart.get_mut(&item) {
                *existing = (*existing).add(value);
            } else {
                let first = first_ys[ys as usize];
                col.waiting_for.entry(first).or_default().push(item);
                col.i_chart.insert(item, value);
            }
        }
    }

    fn predict(&self, col: &mut RescaledColumn) {
        let k = col.k;

        let mut agenda: Vec<u32> = if k == 0 {
            vec![self.start]
        } else {
            col.waiting_for
                .keys()
                .filter(|sym| self.nonterminals.contains(sym))
                .copied()
                .collect()
        };

        let mut reachable: HashSet<u32> = agenda.iter().copied().collect();

        while let Some(x) = agenda.pop() {
            if let Some(targets) = self.outgoing.get(&x) {
                for &y in targets {
                    if reachable.insert(y) {
                        agenda.push(y);
                    }
                }
            }
        }

        for x in &reachable {
            if let Some(rules) = self.rhs.get(x) {
                for &(w, ys) in rules {
                    if ys == 0 {
                        let item = (k, *x);
                        *col.c_chart.entry(item).or_insert(0.0) += w;
                    } else {
                        let item = (k, *x, ys);
                        if let Some(existing) = col.i_chart.get_mut(&item) {
                            *existing = (*existing).add(w);
                        } else {
                            let first = self.first_ys[ys as usize];
                            col.waiting_for.entry(first).or_default().push(item);
                            col.i_chart.insert(item, w);
                        }
                    }
                }
            }
        }
    }

    fn next_token_weights_inner(&self, col_indices: &[usize]) -> HashMap<String, f64> {
        let last_idx = *col_indices.last().unwrap();
        let col = self.col(last_idx);

        let mut q: HashMap<(u32, u32), f64> = HashMap::new();
        q.insert((0, self.start), Semiring::ONE);

        let mut result: HashMap<String, f64> = HashMap::new();

        for (&y_id, items) in &col.waiting_for {
            if !self.id_to_terminal.contains_key(&y_id) {
                continue;
            }

            let mut total = 0.0f64;
            for &(i, x, ys) in items {
                if self.unit_ys[ys as usize] {
                    let node = (i, x);
                    let value = self.helper(node, col_indices, &mut q);
                    total = total.add(col.i_chart[&(i, x, ys)].mul(value));
                }
            }

            if total != 0.0 {
                let terminal_str = &self.id_to_terminal[&y_id];
                result.insert(terminal_str.clone(), total);
            }
        }

        result
    }

    fn helper(
        &self,
        top: (u32, u32),
        col_indices: &[usize],
        q: &mut HashMap<(u32, u32), f64>,
    ) -> f64 {
        if let Some(&v) = q.get(&top) {
            return v;
        }

        struct Frame {
            node: (u32, u32),
            edges: Vec<(u32, u32, u32)>,
            cursor: usize,
            value: f64,
        }

        let mut stack: Vec<Frame> = vec![Frame {
            node: top,
            edges: Vec::new(),
            cursor: usize::MAX,
            value: 0.0,
        }];

        while let Some(frame) = stack.last_mut() {
            let (j, y) = frame.node;

            if frame.cursor == usize::MAX {
                if let Some(&cached) = q.get(&frame.node) {
                    let val = cached;
                    stack.pop();
                    if let Some(parent) = stack.last_mut() {
                        if parent.cursor < parent.edges.len() {
                            let (pi, px, _) = parent.edges[parent.cursor];
                            let col_j_idx = col_indices[j as usize];
                            let iw = self.col(col_j_idx).i_chart
                                .get(&(pi, px, parent.edges[parent.cursor].2))
                                .copied()
                                .unwrap_or(Semiring::ZERO);
                            parent.value = parent.value.add(iw.mul(val));
                            parent.cursor += 1;
                        }
                    }
                    continue;
                }

                let col_j_idx = col_indices[j as usize];
                let col_j = self.col(col_j_idx);
                let edges: Vec<(u32, u32, u32)> = col_j
                    .waiting_for
                    .get(&y)
                    .map(|items| {
                        items.iter()
                            .filter(|(_, _, ys)| self.unit_ys[*ys as usize])
                            .copied()
                            .collect()
                    })
                    .unwrap_or_default();
                frame.edges = edges;
                frame.cursor = 0;
            }

            if frame.cursor >= frame.edges.len() {
                let node = frame.node;
                let value = frame.value;
                q.insert(node, value);
                stack.pop();

                if let Some(parent) = stack.last_mut() {
                    if parent.cursor < parent.edges.len() {
                        let (pi, px, pys) = parent.edges[parent.cursor];
                        let (pj, _py) = parent.node;
                        let col_pj_idx = col_indices[pj as usize];
                        let iw = self.col(col_pj_idx).i_chart
                            .get(&(pi, px, pys))
                            .copied()
                            .unwrap_or(Semiring::ZERO);
                        parent.value = parent.value.add(iw.mul(value));
                        parent.cursor += 1;
                    }
                }
            } else {
                let (ei, ex, _eys) = frame.edges[frame.cursor];
                let neighbor = (ei, ex);

                if let Some(&cached) = q.get(&neighbor) {
                    let (pi, px, pys) = frame.edges[frame.cursor];
                    let col_j_idx = col_indices[j as usize];
                    let iw = self.col(col_j_idx).i_chart
                        .get(&(pi, px, pys))
                        .copied()
                        .unwrap_or(Semiring::ZERO);
                    frame.value = frame.value.add(iw.mul(cached));
                    frame.cursor += 1;
                } else {
                    stack.push(Frame {
                        node: neighbor,
                        edges: Vec::new(),
                        cursor: usize::MAX,
                        value: Semiring::ZERO,
                    });
                }
            }
        }

        q[&top]
    }
}

// ── Python Module ─────────────────────────────────────────────────────────────

#[pymodule]
fn genlm_earley(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustEarley>()?;
    m.add_class::<RustEarleyBool>()?;
    m.add_class::<RustEarleyRescaled>()?;
    Ok(())
}
