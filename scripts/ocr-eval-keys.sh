# shellcheck shell=sh
# On-demand API-key loader for the ocr-eval / realdoc-bench harness.
#
# SOURCE this, never execute it — a child process cannot export into your shell:
#     . ~/bin/ocr-eval-keys.sh      # or the `gemkey` function installed in ~/.zshrc
#
# WHY on-demand rather than a plain `export` in ~/.zshrc: a profile export puts the secret in
# cleartext in a dotfile that gets backed up, synced to dotfile repos, and read into EVERY shell
# you open — including ones running unrelated tools whose crash handlers dump the environment.
# That is exactly how a rotated key survived in ~/.zshrc.bak after the key itself was replaced.
# Here the value rests in one 0600 file and enters only the shell that asks for it.
#
# The harness itself needs nothing from this file: every key is read through a plain
# os.environ.get(), so any mechanism that populates the environment works identically
# (see docs/api.md "Keys" in the ocr-eval repo).

_ocr_eval_secrets="${OCR_EVAL_SECRETS:-$HOME/.config/ocr-eval/secrets.env}"

if [ ! -f "$_ocr_eval_secrets" ]; then
    printf 'ocr-eval-keys: no secrets file at %s\n' "$_ocr_eval_secrets" >&2
    printf '  create it with (umask keeps it 0600, and the value never reaches argv/history):\n' >&2
    printf '    umask 077; printf %s%s%s > %s\n' \
           "'export GEMINI_API_KEY=YOUR_KEY\\n'" "" "" "$_ocr_eval_secrets" >&2
    unset _ocr_eval_secrets
    return 1 2>/dev/null || exit 1
fi

# Refuse to load a file others can read. A silent chmod would hide the fact that the secret was
# already world-readable for some window; failing makes it a decision.
_ocr_eval_perm=$(stat -c '%a' "$_ocr_eval_secrets" 2>/dev/null)
case "$_ocr_eval_perm" in
    600|400) : ;;
    *)
        printf 'ocr-eval-keys: REFUSING to load %s — mode %s, expected 600\n' \
               "$_ocr_eval_secrets" "$_ocr_eval_perm" >&2
        printf '  fix with: chmod 600 %s\n' "$_ocr_eval_secrets" >&2
        unset _ocr_eval_secrets _ocr_eval_perm
        return 1 2>/dev/null || exit 1
        ;;
esac

# shellcheck disable=SC1090
. "$_ocr_eval_secrets"

# Presence check only — NEVER echo a value. `${VAR:+SET}` looks like it prints the literal SET but
# expands to the VALUE itself; that idiom leaked a live key into a session transcript twice.
for _ocr_eval_v in GEMINI_API_KEY OPENROUTER_API_KEY MISTRAL_API_KEY DOCSTRANGE_API_KEY; do
    if [ -n "$(printenv "$_ocr_eval_v")" ]; then
        printf '  %s: loaded\n' "$_ocr_eval_v"
    fi
done

unset _ocr_eval_secrets _ocr_eval_perm _ocr_eval_v
