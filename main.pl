:- use_module(library(ffi)).
:- use_module(library(format)).

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
	collect_case_scores(passing_case, PassingCases),
	collect_case_scores(failing_case, FailingCases),
	print_case_statistics('Should pass', PassingCases),
	nl,
	print_case_statistics('Should fail', FailingCases),
	test_passing_cases(PassingCases, Threshold),
	test_failing_cases(FailingCases, Threshold),
	nl,
	write('All fuzzy-match FFI cases passed.'),
	nl.

collect_case_scores(CasePredicate, Cases) :-
	findall(case(Concept, Target, Score),
		(call(CasePredicate, Concept, Target),
		 fuzzy_score(Concept, Target, Score)),
		Cases).

print_case_statistics(Label, Cases) :-
	case_score_range(Cases, Minimum, Maximum),
	format("~a (min=~4f, max=~4f):~n", [Label, Minimum, Maximum]),
	print_case_scores(Cases).

print_case_scores([]).
print_case_scores([case(Concept, Target, Score)|Cases]) :-
	format("Concept: ~s~t~24| | Target: ~s~t~48| | Spread Score: ~4f~n",
		[Concept, Target, Score]),
	print_case_scores(Cases).

case_score_range([case(_, _, Score)|Cases], Minimum, Maximum) :-
	case_score_range_(Cases, Score, Score, Minimum, Maximum).

case_score_range_([], Minimum, Maximum, Minimum, Maximum).
case_score_range_([case(_, _, Score)|Cases], Minimum0, Maximum0,
				  Minimum, Maximum) :-
	(   Score < Minimum0 -> Minimum1 = Score
	;   Minimum1 = Minimum0
	),
	(   Score > Maximum0 -> Maximum1 = Score
	;   Maximum1 = Maximum0
	),
	case_score_range_(Cases, Minimum1, Maximum1, Minimum, Maximum).

test_passing_cases([], _).
test_passing_cases([case(Concept, Target, Score)|Cases], Threshold) :-
	(   fuzzy_match(Concept, Target, Threshold) ->
		true
	;   format("Expected match failed: ~s - ~s (score=~4f, threshold=~4f)~n",
			[Concept, Target, Score, Threshold]),
		false
	),
	test_passing_cases(Cases, Threshold).

test_failing_cases([], _).
test_failing_cases([case(Concept, Target, Score)|Cases], Threshold) :-
	(   fuzzy_match(Concept, Target, Threshold) ->
		format("Unexpected match succeeded: ~s - ~s (score=~4f, threshold=~4f)~n",
			[Concept, Target, Score, Threshold]),
		false
	;   true
	),
	test_failing_cases(Cases, Threshold).
