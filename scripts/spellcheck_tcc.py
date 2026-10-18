#!/usr/bin/env python3
"""Spellcheck Portuguese TCC .tex sources with hunspell pt_BR."""
import collections
import os
import re
import subprocess
from pathlib import Path

BASE = Path("/Users/vitoriakelly/Documents/TCC/TCC1_extracted")
DICT = Path("/Users/vitoriakelly/Documents/TCC/.spellcheck")

FILES = [
    BASE / "1-pre-textuais/resumo.tex",
    BASE / "1-pre-textuais/lista-abreviaturas-completa.tex",
    BASE / "2-textuais/1-introducao.tex",
    BASE / "2-textuais/2-fundamentacao-teorica.tex",
    BASE / "2-textuais/trabalhos-relacionados.tex",
    BASE / "2-textuais/3-metodologia.tex",
    BASE / "2-textuais/4-resultados.tex",
    BASE / "2-textuais/5-conclusao.tex",
]

# Accepted technical terms, names, English kept in PT chapters
ALLOW = {
    w.lower()
    for w in """
    adc avc bids cad cnn cnns crf dicom dl dsc dwi eda flair ia iou isles miccai ml
    nifti relu rm tc tcc unet u-net dice pixels pixel voxel voxels baseline baselines
    dataset datasets pipeline pipelines framework frameworks benchmark benchmarks
    challenge challenges ground truth overlay overlays threshold thresholds
    preprocessing encoding decoding bottleneck encoder decoder feature features
    kernel kernels pooling stride padding batch normalization dropout softmax
    sigmoid entropy loss optimizer adam sgd epoch epochs augmentation transfer
    fine tuning open source github python numpy scipy nibabel matplotlib pandas
    software hardware online offline heatmap heatmaps patch patches skip connections
    connection attention residual dense transformer transformers morphology
    morphological dilation erosion false true positive negative positives negatives
    precision recall sensitivity specificity accuracy mean median standard deviation
    boxplot histogram qualitative quantitative exploratory analysis registration
    resampling skull stripping bias field correction annotation annotations
    computer aided diagnosis system systems workflow module modules evaluation
    metrics metric limitation limitations contribution contributions related works
    public available multicentric multimodal end state art hiperparâmetros
    hiperintensa hiperintensas hipointensa hipointensas hiperintensidade
    hipointensidade multicêntrico multicêntrica semiautomático semiautomática
    pré-processamento pós-processamento volumetria volumétrico volumétrica
    anisotrópico isotrópico anisotrópica isotrópica coorientador coorientação
    crateús ceará petzsche litjens maier isensee winzeck joshi singh willis
    philips parzen shapes ronneberger kamnitsas deepmedic nnu-net nnunet
    monai pytorch tensorflow keras docker jupyter colab cuda siemens ge
    wellington franco bruno riccelli vitoria vitória kelly overleaf tectonic
    latex bibtex doi arxiv springer elsevier ieee nature plos frontiers who
    abnt ufc htbp iii iv shapes
    """.split()
}


def strip_latex(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(
        r"\\(cite|citep|label|ref|pageref|eqref|gls|glspl|url)\{[^}]*\}", " ", text
    )
    text = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}", " ", text)
    text = re.sub(r"\\begin\{[^}]*\}", " ", text)
    text = re.sub(r"\\end\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = text.replace("{", " ").replace("}", " ").replace("[", " ").replace("]", " ")
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"[0-9]+([.,][0-9]+)?", " ", text)
    text = re.sub(r"[_\\/~|^<>#&@]", " ", text)
    return text


def main() -> None:
    candidates = []
    for f in FILES:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip().startswith("%"):
                continue
            plain = strip_latex(line)
            for w in re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]*", plain):
                if len(w) <= 2:
                    continue
                if w.isupper() and len(w) <= 8:
                    continue
                if w.lower() in ALLOW:
                    continue
                if re.match(r"^[A-Z][a-z]+[A-Z]", w):
                    continue
                candidates.append((w, str(f.relative_to(BASE)), i, line.strip()[:140]))

    unique = sorted({c[0] for c in candidates})
    proc = subprocess.run(
        ["hunspell", "-d", "pt_BR", "-l", "-i", "UTF-8"],
        input="\n".join(unique),
        text=True,
        capture_output=True,
        env={**os.environ, "DICPATH": str(DICT)},
        check=False,
    )
    bad = set(proc.stdout.split())
    bad_l = {b.lower() for b in bad}

    # Extra English leftovers commonly embedded in PT academic text
    eng = {
        w.lower()
        for w in """
        the and of for with from into over under between using based model models
        method methods approach approaches results result performance proposed
        network networks training trained validation test testing score scores
        volume volumes mask masks lesion lesions stroke ischemic patient patients
        clinical case cases available challenge challenges state art deep learning
        machine supervised unsupervised convolutional architecture architectures
        encoder decoder skip connection connections attention residual dense
        transformer augmentation normalization activation function functions layer
        layers output input binary cross entropy dice coefficient intersection union
        precision recall sensitivity specificity accuracy mean median standard
        deviation boxplot histogram qualitative quantitative exploratory analysis
        preprocessing postprocessing morphology morphological dilation erosion
        opening closing connected components thresholding intensity modality
        modalities sequence sequences acquisition scanner resolution anisotropic
        isotropic registration resampling skull stripping bias field correction
        ground truth annotation annotations manual automatic computer aided
        diagnosis system systems workflow module evaluation metrics limitation
        limitations future work contribution contributions objective objectives
        related works theoretical foundation dataset datasets publicly available
        benchmark benchmarks open source repository implementation script scripts
        figure figures table tables chapter sections reference references abstract
        keywords brain cerebral vascular infarct infarction hemorrhage perfusion
        diffusion weighted imaging resonance magnetic computed tomography
        angiography artery arteries penumbra thrombolysis thrombectomy endovascular
        neurology radiology radiologist physician clinician decision support
        diagnosis diagnostic prognosis outcome outcomes survival mortality morbidity
        incidence prevalence epidemiology health organization brazil brazilian
        university federal information systems course undergraduate thesis advisor
        supervisor false true positive negative overlay heatmap baseline feature map
        maps rectified linear unit softmax sigmoid relu conditional random field crf
        neuroimaging informatics technology initiative nifti fluid attenuated
        inversion recovery flair apparent diffusion coefficient adc dwi ischemic
        lesion segmentation isles miccai brain imaging data structure bids digital
        imaging communications medicine dicom machine learning deep learning
        artificial intelligence computer aided diagnosis cad intersection over union
        iou dice similarity coefficient dsc convolutional neural network cnn unet
        exploratory data analysis eda shapes petzsche litjens maier isensee winzeck
        joshi singh willis philips parzen name description template simbolos htbp
        iii pixels hiperparâmetros hiperintensa deepmedic
        """.split()
    }

    final = collections.Counter()
    locs = collections.defaultdict(list)
    for w, path, line, ctx in candidates:
        if w not in bad and w.lower() not in bad_l:
            continue
        wl = w.lower()
        if wl in eng or wl in ALLOW:
            continue
        # proper nouns / author names
        if w[0].isupper() and path.endswith("trabalhos-relacionados.tex"):
            continue
        if w[0].isupper() and w not in {"Brasil", "Figura", "Tabela", "Capítulo"}:
            # keep scanning capitalized mid-sentence PT words carefully:
            # skip likely surnames
            if path.endswith(("3-metodologia.tex", "2-fundamentacao-teorica.tex")):
                if w in {
                    "Philips",
                    "Siemens",
                    "Petzsche",
                    "Ronneberger",
                    "DeepMedic",
                    "Willis",
                    "Shapes",
                    "Parzen",
                    "Maier",
                    "Litjens",
                    "Isensee",
                    "Winzeck",
                    "Joshi",
                    "Singh",
                }:
                    continue
        final[w] += 1
        if len(locs[w]) < 2:
            locs[w].append(f"{path}:{line}: {ctx}")

    print("=== Candidatos a revisão ortográfica (PT) ===\n")
    for w, c in final.most_common(80):
        print(f"{c}x  {w}")
        for L in locs[w]:
            print(f"    {L}")
    print(f"\nTotal formas suspeitas: {len(final)}")


if __name__ == "__main__":
    main()
