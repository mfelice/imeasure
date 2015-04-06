# -*- coding: utf-8 -*-
#!/usr/bin/python

# The MIT License (MIT)
# 
# Copyright (c) 2015 Mariano Felice
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# For a detailed description of the evaluation method implemented in this script,
# refer to the following paper:
#
# Mariano Felice and Ted Briscoe. 2015. Towards a standard evaluation method 
# for grammatical error detection and correction. In Proceedings of the 2015 
# Conference of the North American Chapter of the Association for Computational
# Linguistics: Human Language Technologies (NAACL-HLT 2015), Denver, CO. 
# Association for Computational Linguistics. (To appear)
#
# Please, cite the paper when using this evaluation script in your work.

from collections import Counter
import xml.etree.cElementTree as ET
import candgen as CG
import align
import math
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

### GLOBALS ###

# Constants

SRC  = 0     # Source (writer)
HYP  = 1     # Hypothesis (system)
REF  = 2     # Reference (gold standard)

D    = 'd'   # Detection
C    = 'c'   # Correction

TP   = 'tp'  # True positives
TN   = 'tn'  # True negatives
FP   = 'fp'  # False positives
FN   = 'fn'  # False negatives
FPN  = 'fpn' # Intersection of FPs and FNs (only for correction)

P    = 'p'    # Precision
R    = 'r'    # Recall
F    = 'f'    # F measure
I    = 'i'    # Improvement on WACC
ACC  = 'acc'  # Accuracy
WACC = 'wacc' # Weighted accuracy

CORPUS   = 'corpus' # Optimise at the corpus level
SENTENCE = 'sent'   # Optimise at the sentence level

# Default parameters

file_ref   = None
file_hyp   = None
verbose    = False
v_verbose  = False
mix        = True  # Mix independent corrections
max_a      = C     # Aspect
max_m      = WACC  # Metric
max_metric = (max_a, max_m)
optimise   = SENTENCE
per_sent   = False

b   = 1.0 # Beta for F measure
w   = 2.0 # Weight for weighted accuracy


### FUNCTIONS ###

def get_counts(alignment):
	counts = Counter({ D:Counter({TP:0,TN:0,FP:0,FN:0,FPN:0}), 
					   C:Counter({TP:0,TN:0,FP:0,FN:0,FPN:0}) })
	for i in range(len(alignment[SRC])):
		if alignment[SRC][i] == alignment[HYP][i]:
			if alignment[HYP][i] == alignment[REF][i]:
				counts[D][TN] += 1
				counts[C][TN] += 1
			else:
				if alignment[SRC][i] == alignment[REF][i]:
					counts[D][FP] += 1
					counts[C][FP] += 1
				else:
					counts[D][FN] += 1
					counts[C][FN] += 1
		else:
			if alignment[SRC][i] == alignment[REF][i]:
				counts[D][FP] += 1
				counts[C][FP] += 1
			else:
				counts[D][TP] += 1
				if alignment[HYP][i] == alignment[REF][i]:
					counts[C][TP] += 1
				else:
					counts[C][FP]  += 1
					counts[C][FN]  += 1
					counts[C][FPN] += 1
	return counts

def compare_scores(cur_scores, max_scores, max_a, max_m):
	# Maximise the specified metric and break ties by I, WACC and ACC
	# in that order. If all are identical, check the other aspect.
	other_a = D if max_a == C else C
	return cur_scores[max_a][max_m]   >  max_scores[max_a][max_m]       or \
		  (cur_scores[max_a][max_m]   == max_scores[max_a][max_m]   and \
		   cur_scores[max_a][I]       >  max_scores[max_a][I])          or \
		  (cur_scores[max_a][max_m]   == max_scores[max_a][max_m]   and \
		   cur_scores[max_a][I]       == max_scores[max_a][I]       and \
		   cur_scores[max_a][WACC]    >  max_scores[max_a][WACC])       or \
		  (cur_scores[max_a][max_m]   == max_scores[max_a][max_m]   and \
		   cur_scores[max_a][I]       == max_scores[max_a][I]       and \
		   cur_scores[max_a][WACC]    == max_scores[max_a][WACC]    and \
		   cur_scores[max_a][ACC]     >  max_scores[max_a][ACC])        or \
		  (cur_scores[max_a][max_m]   == max_scores[max_a][max_m]   and \
		   cur_scores[max_a][I]       == max_scores[max_a][I]       and \
		   cur_scores[max_a][WACC]    == max_scores[max_a][WACC]    and \
		   cur_scores[max_a][ACC]     == max_scores[max_a][ACC]     and \
		   cur_scores[other_a][max_m] >  max_scores[other_a][max_m])    or \
		  (cur_scores[max_a][max_m]   == max_scores[max_a][max_m]   and \
		   cur_scores[max_a][I]       == max_scores[max_a][I]       and \
		   cur_scores[max_a][WACC]    == max_scores[max_a][WACC]    and \
		   cur_scores[max_a][ACC]     == max_scores[max_a][ACC]     and \
		   cur_scores[other_a][max_m] == max_scores[other_a][max_m] and \
		   cur_scores[other_a][I]     >  max_scores[other_a][I])        or \
		  (cur_scores[max_a][max_m]   == max_scores[max_a][max_m]   and \
		   cur_scores[max_a][I]       == max_scores[max_a][I]       and \
		   cur_scores[max_a][WACC]    == max_scores[max_a][WACC]    and \
		   cur_scores[max_a][ACC]     == max_scores[max_a][ACC]     and \
		   cur_scores[other_a][max_m] == max_scores[other_a][max_m] and \
		   cur_scores[other_a][I]     == max_scores[other_a][I]     and \
		   cur_scores[other_a][WACC]  >  max_scores[other_a][WACC])     or \
		  (cur_scores[max_a][max_m]   == max_scores[max_a][max_m]   and \
		   cur_scores[max_a][I]       == max_scores[max_a][I]       and \
		   cur_scores[max_a][WACC]    == max_scores[max_a][WACC]    and \
		   cur_scores[max_a][ACC]     == max_scores[max_a][ACC]     and \
		   cur_scores[other_a][max_m] == max_scores[other_a][max_m] and \
		   cur_scores[other_a][I]     == max_scores[other_a][I]     and \
		   cur_scores[other_a][WACC]  == max_scores[other_a][WACC]  and \
		   cur_scores[other_a][ACC]   >  max_scores[other_a][ACC])

def compute_p(tp, fp):
	return float(tp) / (tp + fp) if (tp + fp) > 0 else 1.0

def compute_r(tp, fn):
	return float(tp) / (tp + fn) if (tp + fn) > 0 else 1.0

def compute_f(p, r, b):
	try:
		f = (1 + b**2) * ((p*r) / ((b**2) * p + r))
	except Exception as e:
		f = 0.0
	return f

def compute_acc(tp, tn, fp, fn, fpn):
	return float(tp + tn) / (tp + tn + fp + fn - fpn)

def compute_wacc(tp, tn, fp, fn, fpn, w):
	return float(w * tp + tn) / (w * (tp + fp) + tn + fn - (w + 1) * (float(fpn) / 2))

def compute_i(wacc_sys, wacc_base):
	if wacc_sys == wacc_base:
		return math.floor(wacc_sys)
	elif wacc_sys > wacc_base:
		return (wacc_sys - wacc_base) / (1.0 - wacc_base)
	else:
		return wacc_sys / wacc_base - 1.0

def compute_all(counts_sys, counts_base=None):
	scores_sys = { D:{TP:0,TN:0,FP:0,FN:0,FPN:0,P:0,R:0,F:0,ACC:0,WACC:0,I:0}, 
				   C:{TP:0,TN:0,FP:0,FN:0,FPN:0,P:0,R:0,F:0,ACC:0,WACC:0,I:0} }
	for a in (D,C):
		scores_sys[a].update(counts_sys[a]) # Copy counts
		scores_sys[a][P]    = compute_p   (counts_sys[a][TP], counts_sys[a][FP])
		scores_sys[a][P]    = compute_p   (counts_sys[a][TP], counts_sys[a][FP])
		scores_sys[a][R]    = compute_r   (counts_sys[a][TP], counts_sys[a][FN])
		scores_sys[a][F]    = compute_f   (scores_sys[a][P],  scores_sys[a][R], b)
		scores_sys[a][ACC]  = compute_acc (counts_sys[a][TP], counts_sys[a][TN], 
										   counts_sys[a][FP], counts_sys[a][FN],
									       counts_sys[a][FPN])
		scores_sys[a][WACC] = compute_wacc(counts_sys[a][TP], counts_sys[a][TN], 
									       counts_sys[a][FP], counts_sys[a][FN],
									       counts_sys[a][FPN], w)
		if counts_base:
			wacc_base  = compute_wacc(counts_base[a][TP], counts_base[a][TN], 
									  counts_base[a][FP], counts_base[a][FN],
									  counts_base[a][FPN], w)
			scores_sys[a][I]= compute_i(scores_sys[a][WACC], wacc_base)
	return scores_sys

def get_best_ref_counts(sid, src, hyp, ref_list, cur_sys_counts, cur_base_counts, max_a, max_m, b, w, optimise, per_sent):
	best_ref           = 0
	best_ref_counts    = None
	best_ref_alignment = None
	best_base_counts   = None
	
	max_scores = { D:{TP:-1,TN:-1,FP:-1,FN:-1,P:-1,R:-1,F:-1,ACC:-1,WACC:-1,I:-1}, 
				   C:{TP:-1,TN:-1,FP:-1,FN:-1,P:-1,R:-1,F:-1,ACC:-1,WACC:-1,I:-1} }

	if v_verbose:
		print "Number of references:", len(ref_list)
		
	for i in xrange(len(ref_list)):
		ref = ref_list[i]
		# Get alignment
		ref_alignment = align.Alignment(src, hyp, ref.tokens)
		# Get counts for this reference
		ref_counts = get_counts(ref_alignment.alignment)
		# Get scores for this reference
		ref_scores = compute_all(ref_counts)

		# Compute baseline for this reference
		# Simulate HYP = SRC
		base_alignment = ref_alignment.alignment[:]
		base_alignment[1] = ref_alignment[0][:]
		# Get baseline counts
		base_counts = get_counts(base_alignment)
		# Get baseline scores
		base_scores = compute_all(base_counts)
		# Update reference scores with I measure
		for a in (D,C):
			ref_scores[a][I]= compute_i(ref_scores[a][WACC], base_scores[a][WACC])		

		if v_verbose or optimise == CORPUS:
			# Compute cumulative counts
			cum_sys_counts = cur_sys_counts + ref_counts
			# Compute cumulative scores
			cum_sys_scores = compute_all(cum_sys_counts)
			# Compute cumulative baseline counts
			cum_base_counts = cur_base_counts + base_counts
			# Compute baseline metrics
			cum_base_scores = compute_all(cum_base_counts)
			# Update with I scores
			for a in (D,C):
				cum_sys_scores[a][I]= compute_i(cum_sys_scores[a][WACC], cum_base_scores[a][WACC])

		# Choose scores to compare
		if optimise == CORPUS:
			comparative_scores = cum_sys_scores
		elif optimise == SENTENCE:
			comparative_scores = ref_scores
		
		# Keep this reference if it improves on the maximising metric
		if compare_scores(comparative_scores, max_scores, max_a, max_m):
			# Save new best reference
			best_ref = i
			best_ref_counts = ref_counts
			best_ref_alignment = ref_alignment
			best_base_counts = base_counts
			max_scores = comparative_scores

		if v_verbose:
			print "\nReference:", i, "\n"
			ref_alignment.printme()
			print "\nReference counts:", ref_counts
			print "Reference scores:", ref_scores
			print "Baseline counts:", base_counts
			print "Baseline scores:", base_scores
			print "Cumulative baseline scores:", cum_base_scores
			print "Cumulative system scores:", cum_sys_scores
			print "Maximum system scores:", max_scores
			
	# -- END FOR

	if per_sent or v_verbose:
		# Best reference results
		best_ref_scores = compute_all(best_ref_counts)
		best_base_scores = compute_all(best_base_counts)
		# Update with I scores
		for a in (D,C):
			best_ref_scores[a][I]= compute_i(best_ref_scores[a][WACC], best_base_scores[a][WACC])
		if per_sent:
			print_rows(best_ref_scores, best_base_scores, sid)
		if v_verbose:
			# Best reference results
			print "\nBest reference:", best_ref, "\n"
			best_ref_alignment.printme()
			print "\nCounts:", best_ref_counts
			print "Scores:", best_ref_scores
			print "Baseline counts:", best_base_counts
			print "Baseline scores:", best_base_scores

	return best_ref, best_ref_counts, best_base_counts

def print_info(file_ref, file_hyp, max_a, max_m, optimise, b, w):
	print "---------------------------------------------------------------------------------------------------------------------"
	print "Hypothesis file    :", file_hyp
	print "Gold standard file :", file_ref
	print "Maximising metric  :", max_m.upper(), "-", 'DETECTION' if max_a == D else 'CORRECTION'
	print "Optimise for       :", 'SENTENCE' if optimise == SENTENCE else 'CORPUS'
	print "WAcc weight        :", w
	print "F beta             :", b
	print "---------------------------------------------------------------------------------------------------------------------"

def print_header(per_sent=False):
	cols = "SID     Asp.  " if per_sent else "Aspect        "
	print "---------------------------------------------------------------------------------------------------------------------"
	print cols + "    TP      TN      FP      FN     FPN       P       R  F_{:3.2f}     Acc   Acc_b    WAcc  WAcc_b       I ".format(b)
	print "---------------------------------------------------------------------------------------------------------------------"

def print_rows(ss, sb, sid=None):
	if sid:
		aspect_d = sid.ljust(6) + "  Det "
		aspect_c = sid.ljust(6) + "  Cor "
	else:
		aspect_d = "Detection   "
		aspect_c = "Correction  "
	
	print aspect_d + "  {:6d}  {:6d}  {:6d}  {:6d}  {:6d}  {:6.2f}  {:6.2f}  {:6.2f}  {:6.2f}  {:6.2f}  {:6.2f}  {:6.2f}  {:6.2f}".format(
		   ss[D][TP], ss[D][TN], ss[D][FP], ss[D][FN], ss[D][FPN], 
		   ss[D][P]*100, ss[D][R]*100, ss[D][F]*100, ss[D][ACC]*100, 
		   sb[D][ACC]*100, ss[D][WACC]*100, sb[D][WACC]*100, ss[D][I]*100)
	print aspect_c + "  {:6d}  {:6d}  {:6d}  {:6d}  {:6d}  {:6.2f}  {:6.2f}  {:6.2f}  {:6.2f}  {:6.2f}  {:6.2f}  {:6.2f}  {:6.2f}".format(
		   ss[C][TP], ss[C][TN], ss[C][FP], ss[C][FN], ss[C][FPN], 
		   ss[C][P]*100, ss[C][R]*100, ss[C][F]*100, ss[C][ACC]*100, 
		   sb[C][ACC]*100, ss[C][WACC]*100, sb[C][WACC]*100, ss[C][I]*100)

### MAIN ###

# Help
help_str = \
'''
Usage: python ''' + sys.argv[0] + ''' -ref:<file> -hyp:<file> [-nomix] [-max:<metric>] [-opt:sent|corpus] [-b:<n>] [-w:<n>] [-per-sent] [-v] [-vv]

\t -ref   : XML file containing gold standard annotations.
\t -hyp   : Plain text file containing sentence hypotheses (one per line).
\t -nomix : Do not mix corrections from different annotators; match the best individual reference instead.
\t          By default, the scorer will mix such corrections in order to maximise matches. This option disables 
\t          default behaviour.
\t -max   : Maximise scores for the specified metric: dp, dr, df, dacc, dwacc, di, cp, cr, cf, cacc, cwacc or ci.
\t          Preceding 'd' is for detection, 'c' for correction. Available metrics are: tp (true positives), 
\t          tn (true negatives), fp (false positives), fn (false negatives), p (precision), r (recall), 
\t          f (F measure), acc (accuracy), wacc (weighted accuracy), i (improvement on wacc). Default is ''' + max_a + max_m + '''.
\t -opt   : Optimise scores at the sentence or corpus level. Default is ''' + optimise + '''.
\t -b     : Specify beta for the F measure. Default is ''' + str(b) + '''.
\t -w     : Specify weight of true and false positives for weighted accuracy. Default is ''' + str(w) + '''.
\t -per-sent : Show individual results for each sentence.
\t -v     : Verbose output.
\t -vv    : Very verbose output.
'''

# Read parameters
for i in range(1,len(sys.argv)):
	if sys.argv[i].startswith("-ref:"):
		file_ref = sys.argv[i][5:]
	elif sys.argv[i].startswith("-hyp:"):
		file_hyp = sys.argv[i][5:]
	elif sys.argv[i].startswith("-max:"):
		m = sys.argv[i].lower()
		max_a, max_m = (m[5:6], m[6:]) # Split into aspect and metric
		# Validate
		if max_a not in (D,C) or max_m not in (TP,TN,FP,FN,P,R,F,ACC,WACC,I):
			print "Invalid maximising metric:", max_a, max_m
			max_a, max_m = max_metric
			print "Rolling back to default:", max_a, max_m		
	elif sys.argv[i].startswith("-opt:"):
		opt = sys.argv[i][5:].lower()
		# Validate
		if opt in (SENTENCE, CORPUS):
			optimise = opt
		else:
			print "Invalid optimisation parameter:", opt
			print "Rolling back to default:", optimise
	elif sys.argv[i].startswith("-b:"):
		b = float(sys.argv[i][3:])
	elif sys.argv[i].startswith("-w:"):
		w = float(sys.argv[i][3:])
	elif sys.argv[i] == "-nomix":
		mix = False
	elif sys.argv[i] == "-per-sent":
		per_sent = True
	elif sys.argv[i] == "-vv":
		verbose  = True
		v_verbose = True
	elif sys.argv[i] == "-v":
		verbose = True

# Do we have what we need?
if not file_ref or not file_hyp:
	print help_str
	exit(0)

# Totals
# System
t_counts_sys  = Counter({ D:Counter({TP:0,TN:0,FP:0,FN:0,FPN:0}), 
						  C:Counter({TP:0,TN:0,FP:0,FN:0,FPN:0}) })
# Baseline
t_counts_base = Counter({ D:Counter({TP:0,TN:0,FP:0,FN:0,FPN:0}),
						  C:Counter({TP:0,TN:0,FP:0,FN:0,FPN:0}) })

cg = CG.CandidateGenerator()
f_hyp = open(file_hyp,"r")
context = ET.iterparse(file_ref, events=("start", "end"))
context = iter(context)
event, root = context.next()

print_info(file_ref, file_hyp, max_a, max_m, optimise, b, w)

# Show results per sentence?
if per_sent: 
	print "\nSENTENCE RESULTS"
	print_header(per_sent)

# Read gold standard and process each sentence
for event, elem in context:
	if event == "end":
		if elem.tag == "sentence":
			sid = elem.get("id") # Sentence ID
			if verbose:
				print "\n" + "-"*40
				print "Sentence", sid, ":", elem.find("text").text
				print "-"*40

			# Read hypothesis
			hyp = f_hyp.readline().split()
			src = elem.find("text").text.split()
			# Get all possible valid references
			ref_list = cg.get_candidates(elem, mix)
			# Get the values for the best match among the references
			best_ref, best_ref_counts, best_base_counts = \
			get_best_ref_counts(sid, src, hyp, ref_list, t_counts_sys, t_counts_base, max_a, max_m, b, w, optimise, per_sent)
			# Add to totals
			t_counts_sys  = t_counts_sys + best_ref_counts
			t_counts_base = t_counts_base + best_base_counts
			
			if verbose:
				print "\nCUMULATIVE RESULTS:"
				print "Cumulative system counts:", t_counts_sys
				print "Cumulative system scores:", compute_all(t_counts_sys, t_counts_base)
				print "Cumulative baseline counts:", t_counts_base
				print "Cumulative baseline scores:", compute_all(t_counts_base)
			
			# Free up
			elem.clear()
		elif elem.tag == "script":
			# Free up processed elements
			elem.clear()
			root.clear()
f_hyp.close()

# Compute and show final results
ss = compute_all(t_counts_sys, t_counts_base) # System scores
sb = compute_all(t_counts_base) # Baseline scores

print "\nOVERALL RESULTS"
print_header()
print_rows(ss, sb)
print
