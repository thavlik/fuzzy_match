:- use_module(library(ffi)).

load_fuzzy_match_ffi :-
	use_foreign_module("./fuzzy_match/libfuzzy_match.so", [
		fuzzy_match([cstr, cstr, double], bool),
		fuzzy_score([cstr, cstr], double)
	]).

:- initialization(load_fuzzy_match_ffi).

% Public Prolog wrappers around the generated ffi module predicates.
fuzzy_match(Concept, Target, Threshold) :-
	ffi:fuzzy_match(Concept, Target, Threshold).

fuzzy_score(Concept, Target, Score) :-
	ffi:fuzzy_score(Concept, Target, Score).

% Usage:
% ?- fuzzy_match("eosinophilic", "pink", 0.3).
%    true.
% ?- fuzzy_score("eosinophilic", "pink", Score).
%    Score = 0.999...

threshold(0.3).

passing_case("comma-shaped", "curved rod").
passing_case("eosinophilic", "pink").
passing_case("eosinophilic", "red").
passing_case("eosinophilic", "pinkish red").
passing_case("basophilic", "blue").
passing_case("basophilic", "blue-purple").
passing_case("basophilic", "bluish purple").
passing_case("hypertonic", "tight muscle").
passing_case("pleomorphic", "variable shape").
passing_case("fluke", "trematode").
passing_case("spiral bacilli", "S-shaped bacillus").
passing_case("numerous multicolored bruises",
			 "multiple bruises in various stages of healing").
passing_case("ovoid diplococci in chain-like arrangements",
			 "chains of elliptical-shape spheres").

failing_case("basophilic", "red").
failing_case("basophilic", "pink").
failing_case("basophilic", "pinkish red").
failing_case("hypotonic", "tight muscle").

run_tests :-
	threshold(Threshold),
	test_passing_cases(Threshold),
	test_failing_cases(Threshold),
	write('All fuzzy-match FFI cases passed.'),
	nl.

test_passing_cases(Threshold) :-
	\+ (passing_case(Concept, Target),
		\+ fuzzy_match(Concept, Target, Threshold),
		write('Expected match failed: '),
		write(Concept-Target),
		nl
	).

test_failing_cases(Threshold) :-
	\+ (failing_case(Concept, Target),
		fuzzy_match(Concept, Target, Threshold),
		write('Unexpected match succeeded: '),
		write(Concept-Target),
		nl
	).
