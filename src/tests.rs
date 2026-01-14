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
