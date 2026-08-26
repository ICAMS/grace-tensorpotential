// MathJax configuration and loader for the pymdownx.arithmatex "generic" output.
//
// MathJax is pinned to an exact version and loaded with a Subresource Integrity
// hash, so an altered or compromised CDN payload is rejected by the browser
// instead of being executed (cf. the polyfill.io takeover, commit ce50552).
// MkDocs cannot express an `integrity` attribute in `extra_javascript`, which is
// why the script tag is built here instead of being listed in mkdocs.yml.
//
// To move to another MathJax version, bump MATHJAX_VERSION and regenerate the
// hash from the published npm tarball (jsDelivr serves that file verbatim):
//
//   curl -sO https://registry.npmjs.org/mathjax/-/mathjax-<version>.tgz
//   tar xzf mathjax-<version>.tgz package/es5/tex-mml-chtml.js
//   openssl dgst -sha384 -binary package/es5/tex-mml-chtml.js | openssl base64 -A
//
const MATHJAX_VERSION = "3.2.2";
const MATHJAX_SRI = "sha384-Wuix6BuhrWbjDBs24bXrjf4ZQ5aFeFWBuKkFekO2t8xFU0iNaLQfp2K6/1Nxveei";
const MATHJAX_BASE = "https://cdn.jsdelivr.net/npm/mathjax@" + MATHJAX_VERSION + "/es5";

window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  // Resolve lazily loaded components (fonts, extensions) against the same
  // pinned version instead of inferring the base path at runtime.
  loader: {
    paths: { mathjax: MATHJAX_BASE }
  }
};

const mathJaxScript = document.createElement("script");
mathJaxScript.src = MATHJAX_BASE + "/tex-mml-chtml.js";
mathJaxScript.integrity = MATHJAX_SRI;
mathJaxScript.crossOrigin = "anonymous";
mathJaxScript.async = true;
document.head.appendChild(mathJaxScript);

// Material for MkDocs replaces page content on navigation; re-typeset when it does.
if (typeof document$ !== "undefined") {
  document$.subscribe(function () {
    if (window.MathJax && MathJax.startup && MathJax.typesetPromise) {
      MathJax.startup.output.clearCache();
      MathJax.typesetClear();
      MathJax.texReset();
      MathJax.typesetPromise();
    }
  });
}
