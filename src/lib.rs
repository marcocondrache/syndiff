use std::ops::Range;

use arrayvec::ArrayVec;

use crate::{
    syntax_graph::{SyntaxEdge, SyntaxRoute},
    syntax_tree::{SyntaxHint, SyntaxNode, SyntaxTree},
};

mod syntax_delimiters;
mod syntax_graph;
mod syntax_tree;

pub use syntax_tree::build_tree;

#[derive(Debug, Clone, Copy)]
pub struct SyntaxDiffOptions {
    graph_limit: usize,
}

impl Default for SyntaxDiffOptions {
    fn default() -> Self {
        Self {
            graph_limit: 1_000_000,
        }
    }
}

/// Compute a syntax-aware diff between two `SyntaxTree`s.
///
/// When `lhs_range` and `rhs_range` are provided, the returned ranges are clipped to
/// those bounds and made relative (starting from 0). This is useful when diffing syntax
/// trees that may be larger than the region of interest (e.g., a function containing
/// multiple diff hunks).
pub fn diff_trees(
    lhs_tree: &SyntaxTree,
    rhs_tree: &SyntaxTree,
    lhs_range: Option<Range<usize>>,
    rhs_range: Option<Range<usize>>,
    options: Option<SyntaxDiffOptions>,
) -> Option<(Vec<Range<usize>>, Vec<Range<usize>>)> {
    let options = options.unwrap_or_default();
    let route = syntax_graph::shortest_path(lhs_tree, rhs_tree, options.graph_limit)?;

    let (lhs_ranges, rhs_ranges) = collect_ranges(&route, lhs_tree, rhs_tree, lhs_range, rhs_range);

    Some((merge_ranges(lhs_ranges), merge_ranges(rhs_ranges)))
}

fn collect_ranges(
    route: &SyntaxRoute<'_>,
    lhs_tree: &SyntaxTree,
    rhs_tree: &SyntaxTree,
    lhs_bounds: Option<Range<usize>>,
    rhs_bounds: Option<Range<usize>>,
) -> (Vec<Range<usize>>, Vec<Range<usize>>) {
    let mut lhs_ranges = Vec::default();
    let mut rhs_ranges = Vec::default();

    for path in &route.0 {
        let Some(edge) = path.edge else { continue };
        let Some(vertex) = path.from.as_ref() else {
            continue;
        };

        match edge {
            SyntaxEdge::Replaced { levenshtein_pct } => {
                if let (Some(lhs_node), Some(rhs_node)) = (
                    vertex.lhs.id().map(|id| lhs_tree.get(id)),
                    vertex.rhs.id().map(|id| rhs_tree.get(id)),
                ) {
                    if levenshtein_pct > 20 {
                        if let Some((lhs_replace_ranges, rhs_replace_ranges)) = get_replace_ranges(
                            lhs_node,
                            rhs_node,
                            lhs_bounds.as_ref(),
                            rhs_bounds.as_ref(),
                        ) {
                            lhs_ranges.extend(lhs_replace_ranges);
                            rhs_ranges.extend(rhs_replace_ranges);
                        }
                    } else {
                        lhs_ranges.extend(get_novel_ranges(lhs_node, lhs_bounds.as_ref()));
                        rhs_ranges.extend(get_novel_ranges(rhs_node, rhs_bounds.as_ref()));
                    }
                }
            }
            SyntaxEdge::NovelAtomLHS | SyntaxEdge::EnterNovelDelimiterLHS => {
                if let Some(lhs_node) = vertex.lhs.id().map(|id| lhs_tree.get(id)) {
                    lhs_ranges.extend(get_novel_ranges(lhs_node, lhs_bounds.as_ref()));
                }
            }
            SyntaxEdge::NovelAtomRHS | SyntaxEdge::EnterNovelDelimiterRHS => {
                if let Some(rhs_node) = vertex.rhs.id().map(|id| rhs_tree.get(id)) {
                    rhs_ranges.extend(get_novel_ranges(rhs_node, rhs_bounds.as_ref()));
                }
            }
            SyntaxEdge::EnterNovelDelimiterBoth => {
                if let Some(lhs_node) = vertex.lhs.id().map(|id| lhs_tree.get(id)) {
                    lhs_ranges.extend(get_novel_ranges(lhs_node, lhs_bounds.as_ref()));
                }
                if let Some(rhs_node) = vertex.rhs.id().map(|id| rhs_tree.get(id)) {
                    rhs_ranges.extend(get_novel_ranges(rhs_node, rhs_bounds.as_ref()));
                }
            }
            _ => {}
        }
    }

    (lhs_ranges, rhs_ranges)
}

fn get_novel_ranges(node: &SyntaxNode, bounds: Option<&Range<usize>>) -> ArrayVec<Range<usize>, 2> {
    let mut ranges = ArrayVec::new();

    if node.is_atom() {
        if let Some(r) = adjust_range_to_bounds(node.byte_range.clone(), bounds) {
            ranges.push(r);
        }
    } else {
        if let Some(r) = node
            .open_delimiter_range()
            .and_then(|r| adjust_range_to_bounds(r, bounds))
        {
            ranges.push(r);
        }

        if let Some(r) = node
            .close_delimiter_range()
            .and_then(|r| adjust_range_to_bounds(r, bounds))
        {
            ranges.push(r);
        }
    }

    ranges
}

fn get_replace_ranges(
    lhs_node: &SyntaxNode,
    rhs_node: &SyntaxNode,
    lhs_bounds: Option<&Range<usize>>,
    rhs_bounds: Option<&Range<usize>>,
) -> Option<(Vec<Range<usize>>, Vec<Range<usize>>)> {
    if let (Some(SyntaxHint::Comment(_lhs_comment)), Some(SyntaxHint::Comment(_rhs_comment))) =
        (lhs_node.hint.as_ref(), rhs_node.hint.as_ref())
    {
        // Convert relative ranges to absolute byte positions, then adjust to bounds
        let lhs_offset = lhs_node.byte_range.start;
        let rhs_offset = rhs_node.byte_range.start;

        let lhs_ranges = Vec::<Range<usize>>::default()
            .into_iter()
            .map(|r| (r.start + lhs_offset)..(r.end + lhs_offset))
            .filter_map(|r| adjust_range_to_bounds(r, lhs_bounds))
            .collect();

        let rhs_ranges = Vec::<Range<usize>>::default()
            .into_iter()
            .map(|r| (r.start + rhs_offset)..(r.end + rhs_offset))
            .filter_map(|r| adjust_range_to_bounds(r, rhs_bounds))
            .collect();

        Some((lhs_ranges, rhs_ranges))
    } else {
        None
    }
}

fn merge_ranges(mut ranges: Vec<Range<usize>>) -> Vec<Range<usize>> {
    if ranges.is_empty() {
        return ranges;
    }

    ranges.sort_by_key(|r| r.start);
    let mut merged = vec![ranges[0].clone()];

    for range in ranges.into_iter().skip(1) {
        let last = merged.last_mut().expect("merged is non-empty");
        if range.start <= last.end {
            last.end = last.end.max(range.end);
        } else {
            merged.push(range);
        }
    }

    merged
}

fn adjust_range_to_bounds(
    range: Range<usize>,
    bounds: Option<&Range<usize>>,
) -> Option<Range<usize>> {
    let Some(bounds) = bounds else {
        return Some(range);
    };

    if range.end <= bounds.start || range.start >= bounds.end {
        return None;
    }

    let start = range.start.max(bounds.start) - bounds.start;
    let end = range.end.min(bounds.end) - bounds.start;
    Some(start..end)
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use std::ops::Range;

    use crate::syntax_tree::SyntaxTree;
    use crate::{SyntaxDiffOptions, diff_trees};

    macro_rules! define_parser {
        ($name:ident, $language:expr) => {
            #[allow(dead_code)]
            fn $name(source: &str) -> SyntaxTree<'_> {
                let mut parser = tree_sitter::Parser::new();
                parser
                    .set_language(&$language.into())
                    .expect("failed to set language");
                let tree = parser.parse(source, None).expect("failed to parse");
                crate::syntax_tree::build_tree(tree.walk(), source)
            }
        };
    }

    define_parser!(parse_rust, tree_sitter_rust::LANGUAGE);
    define_parser!(parse_json, tree_sitter_json::LANGUAGE);

    fn mark_text(unmarked_text: &str, ranges: &[Range<usize>]) -> String {
        let mut marked_text = unmarked_text.to_string();
        for range in ranges.iter().rev() {
            marked_text.insert(range.end, '»');
            marked_text.insert(range.start, '«');
        }
        marked_text
    }

    fn unmark_text(marked_text: &str) -> (String, Vec<Range<usize>>) {
        let mut unmarked_text = String::new();
        let mut current_start = None;
        let mut ranges = Vec::default();
        let mut last_end = 0;

        for (marker_index, marker) in marked_text.match_indices(&['«', '»']) {
            unmarked_text.push_str(&marked_text[last_end..marker_index]);
            last_end = marker_index + marker.len();

            match marker {
                "«" => {
                    if current_start.is_some() {
                        panic!("duplicate start marker at index {marker_index}")
                    } else {
                        current_start = Some(unmarked_text.len());
                    }
                }
                "»" => {
                    if let Some(start) = current_start.take() {
                        ranges.push(start..unmarked_text.len());
                    } else {
                        panic!("unexpected end marker at index {marker_index}")
                    }
                }
                _ => unreachable!(),
            }
        }

        unmarked_text.push_str(&marked_text[last_end..]);
        (unmarked_text, ranges)
    }

    #[track_caller]
    fn assert_diff(lhs_marked: &str, rhs_marked: &str, parser: fn(&str) -> SyntaxTree) {
        let (lhs_text, expected_lhs_ranges) = unmark_text(lhs_marked);
        let (rhs_text, expected_rhs_ranges) = unmark_text(rhs_marked);

        let lhs_tree = parser(&lhs_text);
        let rhs_tree = parser(&rhs_text);

        let (lhs_ranges, rhs_ranges) = diff_trees(
            &lhs_tree,
            &rhs_tree,
            None,
            None,
            Some(SyntaxDiffOptions::default()),
        )
        .expect("diff should not exceed graph limit");

        let actual_lhs_marked = mark_text(&lhs_text, &lhs_ranges);
        let actual_rhs_marked = mark_text(&rhs_text, &rhs_ranges);

        assert_eq!(
            lhs_ranges, expected_lhs_ranges,
            "LHS ranges mismatch.\nExpected: {lhs_marked}\nActual:   {actual_lhs_marked}"
        );
        assert_eq!(
            rhs_ranges, expected_rhs_ranges,
            "RHS ranges mismatch.\nExpected: {rhs_marked}\nActual:   {actual_rhs_marked}"
        );
    }

    #[test]
    fn test_diff_trees_identical_json() {
        assert_diff(r#"{"a": 1, "b": 2}"#, r#"{"a": 1, "b": 2}"#, parse_json);
    }

    #[test]
    fn test_diff_trees_changed_value() {
        assert_diff(r#"{"a": «1»}"#, r#"{"a": «2»}"#, parse_json);
    }

    #[test]
    fn test_diff_trees_added_key() {
        assert_diff(r#"{"a": 1}"#, r#"{"a": 1«,» «"b":» «2»}"#, parse_json);
    }

    #[test]
    fn test_diff_trees_removed_key() {
        assert_diff(r#"{"a": 1«,» «"b":» «2»}"#, r#"{"a": 1}"#, parse_json);
    }

    #[test]
    fn test_diff_trees_rust_changed_function_body() {
        assert_diff(
            indoc! {r#"
                    fn main() {
                        println!("«hello»");
                    }
                "#},
            indoc! {r#"
                    fn main() {
                        println!("«world»");
                    }
                "#},
            parse_rust,
        );
    }

    #[test]
    fn test_diff_trees_rust_added_function() {
        assert_diff(
            indoc! {r#"
                    fn foo() {
                        println!("foo");
                    }
                "#},
            indoc! {r#"
                    fn foo() {
                        println!("foo");
                    }

                    «fn» «bar()» «{»
                        «println!("bar");»
                    «}»
                "#},
            parse_rust,
        );
    }

    #[test]
    fn test_diff_trees_rust_changed_function_signature() {
        assert_diff(
            indoc! {r#"
                    fn process(x: i32) -> i32 {
                        x «*» «2»
                    }
                "#},
            indoc! {r#"
                    fn process(x: i32«,» «y:» «i32») -> i32 {
                        x «+» «y»
                    }
                "#},
            parse_rust,
        );
    }

    #[test]
    fn test_diff_trees_rust_struct_field_change() {
        assert_diff(
            indoc! {r#"
                    struct Point {
                        x: f64,
                        y: f64,
                    }
                "#},
            indoc! {r#"
                    struct Point {
                        x: f64,
                        y: f64,
                        «z:» «f64,»
                    }
                "#},
            parse_rust,
        );
    }

    #[test]
    fn test_diff_trees_rust_match_arm_change() {
        assert_diff(
            indoc! {r#"
                    fn classify(n: i32) -> &'static str {
                        match n {
                            0 => "zero",
                            1 => "«one»",
                            _ => "other",
                        }
                    }
                "#},
            indoc! {r#"
                    fn classify(n: i32) -> &'static str {
                        match n {
                            0 => "zero",
                            1 «|» «2» => "«small»",
                            _ => "other",
                        }
                    }
                "#},
            parse_rust,
        );
    }

    #[test]
    fn test_diff_trees_rust_nested_closures() {
        assert_diff(
            indoc! {r#"
                fn main() {
                    let result = items
                        .iter()
                        .map(|x| x * «2»)
                        .filter(|x| x > &«5»)
                        .collect::<Vec<_>>();
                }
            "#},
            indoc! {r#"
                fn main() {
                    let result = items
                        .iter()
                        .map(|x| x * «3»)
                        .filter(|x| x > &«10»)
                        .collect::<Vec<_>>();
                }
            "#},
            parse_rust,
        );
    }

    #[test]
    fn test_diff_trees_rust_impl_with_generics() {
        assert_diff(
            indoc! {r#"
                impl<T: Clone> Container<T> {
                    fn new(value: T) -> Self {
                        Self { value }
                    }
                }
            "#},
            indoc! {r#"
                impl<T: Clone «+» «Send»> Container<T> {
                    fn new(value: T) -> Self {
                        Self { value }
                    }

                    «fn» «get(&self)» «->» «&T» «{»
                        «&self.value»
                    «}»
                }
            "#},
            parse_rust,
        );
    }

    #[test]
    fn test_diff_trees_rust_complex_expression_reorder() {
        assert_diff(
            indoc! {r#"
                fn compute() -> i32 {
                    let a = 1;
                    let b = 2;
                    let c = 3;
                    «a» + b + «c»
                }
            "#},
            indoc! {r#"
                fn compute() -> i32 {
                    let a = 1;
                    let b = 2;
                    let c = 3;
                    «c» + b + «a»
                }
            "#},
            parse_rust,
        );
    }

    #[test]
    fn test_diff_trees_rust_enum_variant_change() {
        assert_diff(
            indoc! {r#"
                enum Message {
                    «Quit»,
                    Move { x: i32, y: i32 },
                    Write(String),
                }

                fn handle(msg: Message) {
                    match msg {
                        Message::«Quit» => println!("«quit»"),
                        Message::Move { x, y } => println!("{x}, {y}"),
                        Message::Write(s) => println!("{s}"),
                    }
                }
            "#},
            indoc! {r#"
                enum Message {
                    «Pause»,
                    Move { x: i32, y: i32 },
                    Write(String),
                }

                fn handle(msg: Message) {
                    match msg {
                        Message::«Pause» => println!("«paused»"),
                        Message::Move { x, y } => println!("{x}, {y}"),
                        Message::Write(s) => println!("{s}"),
                    }
                }
            "#},
            parse_rust,
        );
    }
}
