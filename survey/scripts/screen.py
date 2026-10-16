"""Rule-based title/abstract screening of records_dedup.csv (single screener).

Rules are applied in a fixed order; the first rule that fires sets the reason code.
  E5 outside the date window          E6 not English
  E4 not a peer-reviewed article or full conference paper (preprint, thesis, book,
     book chapter outside a proceedings series, erratum/retraction, abstract-only venue)
  E1 not metal additive manufacturing (non-metal or generic manufacturing)
  E3 review/survey/editorial/perspective (routed to the review list)
  E2 no implemented AI/ML model            E7 simulation-only without ML
  E8 insufficient information in title/abstract (full-text check, counted separately),
     including records whose only metal-AM evidence is a generic mention in the abstract
  I1 primary empirical metal-AM ML study
Manual decisions recorded in screening_overrides.csv (ID, reason_code, justification)
replace the rule decision for that ID and are marked as such in the ledger.
Writes screening_ledger.csv.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, WINDOW_END, WINDOW_START, read_csv, write_csv  # noqa: E402

I = re.I

# ---- metal additive manufacturing -------------------------------------------------
FAMILY_PATTERNS = [
    ("LPBF", re.compile(
        r"laser[- ]?powder[- ]?bed[- ]?fusion|selective[- ]laser[- ]melt|direct metal laser (sintering|melting)"
        r"|laser beam melting|powder[- ]bed fusion[- ]?(of metals[- ])?(-|with |using )?laser|pbf[- ]lb"
        r"|laser[- ]based powder[- ]bed|laser powder[- ]bed|metal powder[- ]bed fusion", I),
     re.compile(r"\b(L-?PBF|LPBF|SLM|DMLS|DMLM|PBF-LB(/M)?|PBF-L)\b")),
    ("EB-PBF", re.compile(
        r"electron[- ]beam melt|electron[- ]beam powder[- ]bed|powder[- ]bed fusion[- ]?(-|with |using )?electron"
        r"|pbf[- ]eb|electron[- ]beam[- ]based powder|electron[- ]beam selective melt", I),
     re.compile(r"\b(EBM|EB-?PBF|PBF-EB(/M)?|EBSM)\b")),
    ("DED", re.compile(
        r"direct(ed)?[- ]energy[- ]deposition|laser[- ]metal[- ]deposition|laser engineered net shap"
        r"|laser direct(ed)? (metal )?deposition|direct (laser|metal) deposition|laser melting deposition"
        r"|laser solid forming|powder[- ]fed laser|wire[- ]fed laser|laser[- ]wire (additive|deposition|directed)"
        r"|laser[- ]directed energy|electron[- ]beam freeform|electron[- ]beam (wire|directed energy)"
        r"|laser deposition", I),
     re.compile(r"\b(DED|DED-L[BPW]?|L-DED|LDED|LMD|LMD-[pPwW]|LENS|DLD|DMD|LSF|EBF3|EBAM)\b")),
    ("WAAM", re.compile(
        r"wire[- ]?(and[- ]|\+[- ]?)?arc[- ]?(additive|based additive|directed|deposition|am\b)|arc[- ](based )?additive"
        r"|arc[- ]directed energy deposition|wire arc|(gas metal arc|gas tungsten arc|plasma arc|cold metal transfer"
        r"|metal inert gas|tungsten inert gas)[^.]{0,80}additive|additive[^.]{0,80}(gas metal arc|gas tungsten arc"
        r"|plasma arc|cold metal transfer)", I),
     re.compile(r"\b(WAAM|WA-DED|DED-[Aa]rc|[Aa]rc-DED|PAAM|GMA-AM|GTA-AM)\b")),
    ("BJ", re.compile(r"binder[- ]?jet", I), None),
    ("MEX", re.compile(
        r"bound[- ]metal deposition|metal fused filament|atomic diffusion additive|metal[- ](fff|fdm|mex)\b"
        r"|(metal|stainless steel|copper|titanium|17-4)[^.]{0,40}(fused filament|material extrusion|filament extrusion)", I),
     None),
    ("PBF-unspecified", re.compile(r"powder[- ]bed[- ]fusion|powder[- ]bed", I), re.compile(r"\bPBF\b")),
    ("Other-metal-AM", re.compile(
        r"cold spray additive|additive friction stir|friction stir (additive|deposition)|ultrasonic additive"
        r"|laser cladding|laser additive manufactur", I), None),
]
GENERIC_METAL_AM = re.compile(
    r"metal(lic)?[- ]additive(ly)?[- ]manufactur|metal(lic)? 3d[- ]print|metal[- ]am\b|metal[- ]based additive"
    r"|additive(ly)?[- ]manufactur\w*[^.]{0,60}(metal|alloy|steel|titanium|ti-?6al-?4v|inconel|nickel|aluminium|aluminum|copper|superalloy)"
    r"|(metal|alloy|steel|titanium|ti-?6al-?4v|inconel|superalloy)[^.]{0,60}additive(ly)?[- ]manufactur"
    r"|3d[- ]print\w*[^.]{0,40}(metal|alloy|steel|titanium)", I)
MATERIAL = re.compile(
    r"\bmetal|alloy|steel|ti-?6al-?4v|\bti64\b|titanium|inconel|\bin ?718\b|\bin ?625\b|\bin ?738|alsi|nickel"
    r"|superalloy|aluminium|aluminum|copper|cobalt|cocr|co-cr|maraging|17-4|316l|\b316\b|\b304l?\b|aisi|tungsten"
    r"|magnesium|hastelloy|haynes|nitinol|\bniti\b|tantalum|molybdenum|high[- ]entropy|melt[- ]?pool|keyhole"
    r"|spatter|lack[- ]of[- ]fusion|\bh13\b|invar|bronze|zirconium|ti-\d|\bti\b", I)
AM_GENERIC = re.compile(r"additive(ly)?[- ]manufactur|3d[- ]print|three[- ]dimensional print|layer[- ]by[- ]layer"
                        r"|layer[- ]wise|powder[- ]bed|rapid prototyp", I)
NONMETAL = re.compile(
    r"polymer|polyamide|\bpa ?1[12]\b|nylon|\bpeek\b|thermoplastic|fused deposition model|fused filament fabrication"
    r"|\bfdm\b|\bfff\b|stereolithograph|\bsla\b|vat photopolymer|photopolymer|\bresin|concrete|cement|mortar"
    r"|ceramic|bioprint|bio-print|hydrogel|tissue engineering|\bpla\b|polylactic|\babs\b|acrylonitrile|\bfood"
    r"|sand (mold|mould|core)|\bglass\b|\bclay\b|polyjet|material jetting|selective laser sintering|\bsls\b"
    r"|multi[- ]jet fusion|\bmjf\b|fibre[- ]reinforced|fiber[- ]reinforced|elastomer|\btpu\b|construction printing", I)

# ---- AI / ML --------------------------------------------------------------------
ML_STRONG = re.compile(
    r"machine[- ]learn|deep[- ]learn|neural[- ]net|\bann\b|\banns\b|\bcnns?\b|convolutional|\blstm|long short[- ]term memory"
    r"|recurrent neural|\brnn|\bgru\b|transformer[- ]based|vision transformer|\btransformers?\b(?! oil)|autoencoder|auto-encoder"
    r"|random[- ]forest|gradient[- ]boost|xgboost|lightgbm|catboost|adaboost|support[- ]vector|\bsvms?\b|\bsvr\b"
    r"|gaussian[- ]process|kriging|bayesian optimi[sz]|bayesian neural|bayesian network|bayesian deep|\bgpr\b"
    r"|nearest[- ]neighbo|\bk-?nn\b|decision tree|reinforcement learning|transfer learning|domain adaptation"
    r"|physics[- ]informed (neural|machine|deep|learning|data)|\bpinns?\b|u-?net\b|yolo|resnet|\bvgg|mobilenet"
    r"|efficientnet|densenet|generative adversarial|\bgans?\b|k-means|self-organi[sz]ing map|multi-?layer perceptron"
    r"|\bmlps?\b|extreme learning machine|deep belief|graph neural|\bvit\b|naive bayes|logistic regression"
    r"|ensemble learning|supervised learning|unsupervised|semi-supervised|self-supervised|few-shot|meta-learning"
    r"|active learning|contrastive learning|isolation forest|one-class|dbscan|gaussian mixture|hidden markov"
    r"|learning[- ]based|neural operator|deeponet|diffusion model|large language model|foundation model"
    r"|segment anything|r-cnn|object detection|semantic segmentation|instance segmentation|classifier"
    r"|regression tree|symbolic regression|kernel ridge|\bsindy\b|sparse identification|\bknn\b|\bdnns?\b"
    r"|\bbpnn\b|back[- ]?propagation|\bpca-|dictionary learning|anomaly detection (model|algorithm)"
    r"|multi-?task learning|deep reinforcement|machine[- ]learned|learning model"
    r"|separable convolution|depthwise|convolution (layer|kernel|operation)|feature pyramid|attention mechanism", I)
ML_WEAK = re.compile(
    r"artificial intelligence|\bai\b|data[- ]driven|surrogate model|metamodel|meta-model|computer vision"
    r"|anomaly detection|bayesian|polynomial chaos|principal component|\bpca\b|clustering|classification"
    r"|feature extraction|pattern recognition|digital twin|statistical learning|regression model|intelligent", I)
FUTURE = re.compile(r"future|could|potential|promising|outlook|further work|will enable|may be|might", I)
SIM = re.compile(r"finite[- ]element|\bfem\b|\bfea\b|simulation|numerical model|computational fluid|\bcfd\b"
                 r"|thermal model|phase[- ]field|cellular automat|molecular dynamics|multi-?physics|lattice boltzmann"
                 r"|monte carlo|analytical model|computational model", I)

# ---- article type -----------------------------------------------------------------
REVIEW_TITLE = re.compile(
    r"\breviews?\b|\bsurvey\b|\boverview\b|state[- ]of[- ]the[- ]art|\bperspectives?\b|\beditorial\b|\btutorial\b"
    r"|roadmap|recent (progress|advances|developments|trends)|progress (in|of|on|and)\b|advances (in|of|on)\b"
    r"|developments in|challenges and (opportunities|prospects|future|outlook)|opportunities and challenges"
    r"|current status|status and|trends (in|of|and)|bibliometric|\bliterature\b|\bpreface\b|special issue"
    r"|a critical (look|assessment|appraisal)|\boutlook\b|\bprospects\b|\bprimer\b|lessons learned|\bvision\b paper", I)
REVIEW_ABS = re.compile(
    r"this (review|survey|overview|mini-review)|(paper|article|work|study|chapter) (reviews|surveys)\b|\bwe review"
    r"|\bwe survey|\b(is|are|been) reviewed|comprehensive review|systematic review|literature review|critical review"
    r"|state[- ]of[- ]the[- ]art review|(paper|article|chapter) (summari[sz]es|provides an overview|presents an overview"
    r"|presents a review|presents a comprehensive|gives an overview)|review of (the )?(recent|current|existing|latest)"
    r"|reviews? (the )?(recent|current|latest) (advances|progress|developments|literature|research)"
    r"|we summari[sz]e (the )?(recent|current|existing)|an overview of (the )?(recent|current|state)", I)
PRIMARY_CUES = re.compile(r"\bwe (propose|develop|present|train|demonstrate|introduce|design|build)|(is|was|were) (proposed|developed|trained)"
                          r"|this (paper|study|work|article) (proposes|develops|presents a (novel|new))|experiment"
                          r"|accuracy of|outperform|r2|r\^2|coefficient of determination", I)
EDITORIAL = re.compile(r"^(editorial|guest editorial|preface|foreword|introduction to the special issue)\b", I)
NOT_ARTICLE_TITLE = re.compile(r"^(erratum|corrigendum|correction to|correction:|retraction|retracted|withdrawn)\b", I)
PROCEEDINGS_VOLUME = re.compile(r"proceedings,? part [ivx\d]+\b|^proceedings of\b", I)
E4_CR_TYPES = {"posted-content", "dissertation", "book", "monograph", "edited-book", "report", "report-component",
               "dataset", "standard", "peer-review", "component", "other", "journal-issue", "proceedings",
               "reference-entry", "book-part", "book-section", "book-series", "book-set", "reference-book",
               "grant", "database", "book-track", "proceedings-series", "report-series", "journal"}
PROCEEDINGS_VENUE = re.compile(r"proceedings|conference|symposium|lecture notes|congress|workshop|\bifac\b|cirp|\bsffs\b|solid freeform", I)
ABSTRACT_ONLY_VENUE = re.compile(r"meeting abstracts|bulletin of the american physical society|microscopy and microanalysis"
                                 r"|abstract book|book of abstracts", I)
PREPRINT_VENUE = re.compile(r"arxiv|ssrn|research square|preprints|techrxiv|engrxiv|chemrxiv|authorea", I)
NON_LATIN = re.compile(r"[Ѐ-ӿ؀-ۿ぀-ヿ㐀-鿿가-힯]")
FOREIGN_WORDS = re.compile(r"\b(de|la|del|und|der|die|das|para|em|com|por|des|et|le|les|für|mit|na|da|do|dos|das|zur|von|y|e)\b", I)
EN_WORDS = re.compile(r"\b(the|of|and|for|with|in|on|by|using|based|via|from|to)\b", I)


def families(text):
    out = []
    for name, ci, cs in FAMILY_PATTERNS:
        hit = ci.search(text) or (cs.search(text) if cs else None)
        if not hit:
            continue
        if name == "Other-metal-AM" and "laser cladding" in hit.group(0).lower() and not AM_GENERIC.search(text):
            continue
        out.append(name)
    if "PBF-unspecified" in out and ("LPBF" in out or "EB-PBF" in out):
        out.remove("PBF-unspecified")
    return out


def metal_status(text):
    """Return (is_metal_am, families, evidence)."""
    fams = families(text)
    specific = [f for f in fams if f in ("LPBF", "EB-PBF", "DED", "WAAM", "MEX", "Other-metal-AM")]
    generic = GENERIC_METAL_AM.search(text)
    mat = MATERIAL.search(text)
    if specific:
        return True, fams, "process: " + ",".join(specific)
    if generic:
        return True, fams + ["Metal-AM-unspecified"], "metal AM: '%s'" % generic.group(0)[:40]
    if ("PBF-unspecified" in fams or "BJ" in fams) and mat:
        return True, fams, "%s with metal indicator '%s'" % ("/".join(fams), mat.group(0))
    return False, fams, ""


def ml_status(title, abstract):
    text = title + " " + abstract
    strong = sorted({m.group(0).lower() for m in ML_STRONG.finditer(text)})
    weak = sorted({m.group(0).lower() for m in ML_WEAK.finditer(text)})
    prospect_only = False
    if strong and not ML_STRONG.search(title) and abstract:
        sents = [s for s in re.split(r"(?<=[.!?])\s+", abstract) if ML_STRONG.search(s)]
        prospect_only = bool(sents) and all(FUTURE.search(s) for s in sents)
    return strong, weak, prospect_only


def is_english(title):
    if NON_LATIN.search(title):
        return False
    f = len(FOREIGN_WORDS.findall(title))
    e = len(EN_WORDS.findall(title))
    return not (f >= 2 and e == 0)


def screen(r):
    title, abstract = r["title"] or "", r["abstract"] or ""
    text = title + " " + abstract
    basis = "title+abstract" if abstract else "title only"
    year = int(r["year"]) if r["year"] else None
    pdate = r["publication_date"] or ""
    venue = r["venue"] or ""
    crt = r["crossref_type"] or ""
    s2t = r["s2_publication_types"] or ""

    if (year and (year < 2015 or year > 2026)) or (pdate and (pdate < WINDOW_START or pdate > WINDOW_END)):
        return "E5", "publication date %s outside 2015-01-01..2026-10-15" % (pdate or year), basis
    if not is_english(title):
        return "E6", "title not in English", basis
    if NOT_ARTICLE_TITLE.search(title):
        return "E4", "erratum/correction/retraction notice", basis
    if PROCEEDINGS_VOLUME.search(title):
        return "E4", "whole proceedings volume, not an article", basis
    if r["is_preprint"] == "1":
        return "E4", "preprint without a peer-reviewed version in the retrieved set", basis
    if crt in E4_CR_TYPES:
        return "E4", "Crossref type '%s'" % crt, basis
    if crt == "book-chapter" and not (PROCEEDINGS_VENUE.search(venue) or "Conference" in s2t):
        return "E4", "book chapter outside a proceedings series", basis
    if ABSTRACT_ONLY_VENUE.search(venue):
        return "E4", "abstract-only venue (%s)" % venue[:40], basis
    if not r["doi"] and PREPRINT_VENUE.search(venue):
        return "E4", "no DOI; preprint server record (%s)" % venue[:30], basis
    code, why = screen_content(r, title, abstract, s2t)
    if code in ("I1", "E3") and not r["doi"] and not venue:
        return "E8", "no DOI and no venue; publication status needs full-text check", basis
    return code, why, basis


def screen_content(r, title, abstract, s2t):
    text = title + " " + abstract

    is_metal, fams, mev = metal_status(text)
    nonmetal = NONMETAL.search(text)
    am = AM_GENERIC.search(text) or fams
    is_review = bool(EDITORIAL.search(title) or REVIEW_TITLE.search(title) or REVIEW_ABS.search(abstract)
                     or ("Review" in s2t and len(PRIMARY_CUES.findall(abstract)) < 2))
    if not is_metal:
        if nonmetal and not MATERIAL.search(text):
            return "E1", "non-metal AM/material ('%s'), no metal process or material" % nonmetal.group(0)
        if not abstract:
            if nonmetal:
                return "E1", "non-metal indicator in title ('%s')" % nonmetal.group(0)
            return "E8", "title only; metal AM process not identifiable without abstract"
        if not am:
            return "E1", "not additive manufacturing (no AM process in title/abstract)"
        if nonmetal:
            return "E1", "mixed/non-metal AM ('%s'); no metal-specific process" % nonmetal.group(0)
        if "PBF-unspecified" in fams or "BJ" in fams:
            return "E8", "%s without a stated material; metal or polymer to be checked in full text" % "/".join(fams)
        return "E1", "generic AM or manufacturing; no metal process or metal material stated"

    strong, weak, prospect_only = ml_status(title, abstract)
    title_am = AM_GENERIC.search(title) or families(title) or re.search(r"melt[- ]?pool|print", title, I) or re.search(r"\bAM\b", title)
    if not is_review and "Metal-AM-unspecified" in fams and len(fams) == 1 and not title_am:
        return "E8", "metal AM named only in the abstract; whether the ML is applied to an AM process needs full-text check"
    if is_review:
        why = "editorial/preface" if EDITORIAL.search(title) else "review/survey/perspective"
        return "E3", "%s on metal AM (%s); routed to review list" % (why, mev)
    if strong and not prospect_only:
        return "I1", "metal AM [%s] with ML [%s]" % (";".join(fams) or "metal", ", ".join(strong[:4]))
    if strong and prospect_only:
        return "E8", "ML mentioned only as prospect/future work; check full text"
    if weak:
        return "E8", "only generic AI/data-driven wording [%s]; ML model not identifiable" % ", ".join(weak[:3])
    if not abstract:
        return "E8", "title only; no ML term in title"
    if SIM.search(text):
        return "E7", "simulation/modelling without ML (%s)" % SIM.search(text).group(0)
    return "E2", "metal AM study without an implemented AI/ML model"


def main():
    recs = read_csv(os.path.join(OUT, "records_dedup.csv"))
    if not recs or "abstract" not in recs[0]:
        sys.exit("records_dedup.csv without abstracts; screening needs the output of a new search")
    ovp = os.path.join(OUT, "screening_overrides.csv")
    overrides = {o["ID"]: o for o in read_csv(ovp)} if os.path.exists(ovp) else {}
    rows = []
    for r in recs:
        code, why, basis = screen(r)
        rule_code = code
        src = "rule"
        if r["ID"] in overrides:
            o = overrides[r["ID"]]
            code, why, src = o["reason_code"], o["justification"], "manual override (rule gave %s)" % rule_code
        text = (r["title"] or "") + " " + (r["abstract"] or "")
        _, fams, _ = metal_status(text)
        decision = {"I1": "include", "E3": "exclude (review list)", "E8": "exclude pending full-text check"}.get(code, "exclude")
        rows.append({"ID": r["ID"], "streams": r["streams"], "decision": decision, "reason_code": code,
                     "justification": why, "screening_basis": basis, "decision_source": src,
                     "rule_code": rule_code, "process_families": ";".join(fams), "year": r["year"],
                     "title": r["title"], "doi": r["doi"]})
    write_csv(os.path.join(OUT, "screening_ledger.csv"), rows,
              ["ID", "streams", "decision", "reason_code", "justification", "screening_basis",
               "decision_source", "rule_code", "process_families", "year", "doi", "title"])
    from collections import Counter
    print(Counter(r["reason_code"] for r in rows))


if __name__ == "__main__":
    main()
