import sys
import os

# --- (Keep your DLL loading section at the top if it helped before) ---
# Attempt to load DLL explicitly
# try:
#     from ctypes import CDLL
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     CDLL(os.path.join(script_dir, "libyices.dll"))
#     # Or use a more robust path detection for the DLL
# except Exception as e_dll:
#     print(f"Note: ctypes explicit DLL load attempt failed or skipped: {e_dll}")


try:
    import yices
    # Let's see what version we're dealing with or what attributes it has.
    print(f"Successfully imported 'yices' module.")
    if hasattr(yices, 'YICES_VERSION'):
        print(f"  Yices Version (from wrapper): {yices.YICES_VERSION}")
    if hasattr(yices, '__version__'):  # Wrapper's own version
        print(f"  Yices Python Wrapper Version: {yices.__version__}")
    if hasattr(yices, '__file__'):
        print(f"  Yices Python Wrapper Location: {yices.__file__}")

except ImportError:
    print("CRITICAL: Yices Python wrapper not found or could not be imported.")
    print("Ensure 'yices-smt' package is installed (e.g., pip install yices-smt).")
    print("Also, ensure the Yices C library (libyices.dll / .so) is in your system PATH or accessible.")
    sys.exit(1)
except Exception as e_imp:
    print(f"An unexpected error occurred during Yices import: {e_imp}")
    sys.exit(1)


# --- LCG and Problem Parameters ---
A_val = 68909602460261
M_val = 2**48
MOD_TARGET_val = 47
VAL_TARGET_val = 42
VAL_Q_TARGET_val = 46
N_BITS = 48
INTERMEDIATE_N_BITS = N_BITS * 2
Q_WIDTH = A_val.bit_length()


def bv_val(width, value):
    """Helper to create a Yices bit-vector constant from a Python integer.
       Focuses on yices.parse_term which is the standard Yices 2 way to make terms from strings.
    """
    # Yices 2 standard: create bit-vectors using "0b..." or "0x..." string formats
    # and yices.parse_term or yices.parse_bv (if available & simpler)
    # For numbers > 63 bits, a string representation is almost always required.

    if not hasattr(yices, 'parse_term'):
        raise AttributeError(
            "The Yices Python wrapper is missing the fundamental 'yices.parse_term' function.")

    # We need to provide a width specifier for Yices parser.
    # Option 1: If yices.parse_bv is available and handles width explicitly.
    # Some wrappers expose `yices.parse_bv(width, string_value_without_prefix)` or similar
    # but yices.parse_term is more general for "(mk-bv <val> <width>)" SMT-LIB like style or directly '0b...' etc.

    # Let's use the direct bit-string representation (e.g., "0b001010")
    # which is universally understood by Yices term parsers.

    # Ensure the value isn't negative for binary representation, though these constants are positive.
    if value < 0:
        raise ValueError(
            "Bit-vector values must be non-negative for this simple binary conversion.")

    binary_string_value = bin(value)[2:]  # Get "101010" from "0b101010"

    # Pad with leading zeros to match the required width
    if len(binary_string_value) > width:
        raise ValueError(
            f"Value {value} (binary: {binary_string_value}) is too large to fit in {width} bits.")

    padded_binary_string = binary_string_value.zfill(width)

    # Create the Yices-parseable string: "0b<padded_binary_string>"
    yices_bv_string = "0b" + padded_binary_string

    try:
        term = yices.parse_term(yices_bv_string)
        # We need to verify the term created is indeed of the correct width.
        # yices.type_of_term(term) then yices.is_bv_type(), yices.bv_type_size()
        # For simplicity, we'll assume parse_term creates it correctly from "0b..."
        return term
    except Exception as e:
        # This might happen if parse_term is not handling "0b" as expected without context,
        # or if the yices module itself is not fully initialized/functional.
        # A more complex SMT-LIB string might be needed for parse_term in some wrappers,
        # e.g., parse_term(f"(mk-bv {value} {width})") -- but this makes `value` a decimal string.
        print(f"Error using yices.parse_term('{yices_bv_string}'): {e}")
        # Fallback: try constructing "(mk-bv value width)" style if parse_term can handle it
        # This makes value a decimal, which is fine for mk-bv SMT-LIB constructor
        smtlib_bv_constructor_string = f"(mk-bv {value} {width})"
        try:
            print(
                f"  Attempting fallback: yices.parse_term('{smtlib_bv_constructor_string}')")
            term = yices.parse_term(smtlib_bv_constructor_string)
            return term
        except Exception as e2:
            print(
                f"Error using yices.parse_term with SMT-LIB constructor style: {e2}")
            raise AttributeError(
                "Failed to create bit-vector constant using yices.parse_term with both '0b...' "
                f"and '(mk-bv ...)' styles. Wrapper module: {yices.__file__ if hasattr(yices, '__file__') else 'Unknown yices module'}"
            )


# --- main() function ---
# (Make sure to adapt any generic yices.eq to yices.bv_eq if issues arise,
# and check yices.extract (or yices.bvextract) argument order: (term, low_idx, high_idx) is C API typical.
# Yices term API also might use simplified function names like yices.bvmul (not yices.yices_bvmul for instance).

# --- (Paste your full main() function here, as corrected in previous versions,
# specifically with yices.bv_eq, and carefully check the yices.extract argument order.)
# For example, yices.bvextract(term, 0, N_BITS-1) should be correct.
def main():
    # Initialize Yices (if required by the wrapper)
    if hasattr(yices, 'yices_init'):  # yices-smt package uses this
        yices.yices_init()
    elif hasattr(yices, 'init'):
        yices.init()

    print("Creating constants...")
    try:
        A_val_term_intermediate = bv_val(INTERMEDIATE_N_BITS, A_val)
        M_lcg_term_intermediate_for_Q = bv_val(INTERMEDIATE_N_BITS, M_val)
        mod_target_term_for_Xj_cond = bv_val(N_BITS, MOD_TARGET_val)
        val_target_term_for_Xj_cond = bv_val(N_BITS, VAL_TARGET_val)
        mod_target_term_for_Qj_cond = bv_val(Q_WIDTH, MOD_TARGET_val)
        val_q_target_term_for_Qj_cond = bv_val(Q_WIDTH, VAL_Q_TARGET_val)
        print("Constants created successfully.")
    except AttributeError as e:
        print(f"CRITICAL ERROR DURING CONSTANT CREATION: {e}")
        if hasattr(yices, 'yices_exit'):
            yices.yices_exit()
        elif hasattr(yices, 'exit'):
            yices.exit()
        return  # Stop execution
    except Exception as e_const:
        print(f"UNEXPECTED ERROR DURING CONSTANT CREATION: {e_const}")
        if hasattr(yices, 'yices_exit'):
            yices.yices_exit()
        elif hasattr(yices, 'exit'):
            yices.exit()
        return

    max_k_found = -1
    X0_for_max_k = None
    PRACTICAL_K_LIMIT = 5  # Reduced for faster testing of basic functionality

    cfg = None  # Not strictly needed if not setting specific configurations for this problem
    # if hasattr(yices, 'yices_new_config'): cfg = yices.yices_new_config()

    for k_val_problem in range(PRACTICAL_K_LIMIT):
        print(f"Attempting to find X_0 for k = {k_val_problem} with Yices...")

        ctx = None
        if hasattr(yices, 'yices_new_context'):
            # cfg can be None for default config
            ctx = yices.yices_new_context(cfg)
        else:
            print(
                "Could not create Yices context. API function yices_new_context missing.")
            break

        # X0_type: Yices types are usually implicitly handled by operations or when defining vars.
        # In yices-smt, `new_variable` or `new_uninterpreted_term` creates a named term.
        # We need a fresh variable (term) for X0.
        # Using `yices.new_variable_from_type` or `yices.new_uninterpreted_term` if that is better
        if hasattr(yices, 'new_uninterpreted_term') and hasattr(yices, 'mk_bv_type'):
            bv_type_term = yices.mk_bv_type(N_BITS)
            X0_term = yices.new_uninterpreted_term(
                bv_type_term, b"X0")  # Name usually needs to be bytes
        elif hasattr(yices, 'new_variable'):  # common in yices-smt package
            X0_term = yices.new_variable(
                N_BITS, b"X0")  # name as bytes literal
        else:
            print("Cannot create Yices variable for X0.")
            break

        current_X_term = X0_term

        # Yices function names for operations often follow SMT-LIB closely without 'yices_' prefix
        # (e.g. yices.bvmul, yices.bvurem) in the 'yices-smt' package

        for j_idx in range(k_val_problem + 1):
            if j_idx == 0:
                rem_X0_expr = yices.bvurem(
                    current_X_term, mod_target_term_for_Xj_cond)
                # eq_X0_expr = yices.eq(rem_X0_expr, val_target_term_for_Xj_cond) # General eq
                # Specific for bitvectors
                eq_X0_expr = yices.bveq(
                    rem_X0_expr, val_target_term_for_Xj_cond)
                yices.assert_formula(ctx, eq_X0_expr)
            else:
                # The yices-smt Python package generally uses short names e.g., yices.zero_extend
                if hasattr(yices, 'zero_extend'):
                    X_prev_extended = yices.zero_extend(
                        current_X_term, INTERMEDIATE_N_BITS - N_BITS)
                else:  # fallback to bvzero_extend if the alias isn't there
                    X_prev_extended = yices.bvzero_extend(
                        current_X_term, INTERMEDIATE_N_BITS - N_BITS)

                product_AXprev = yices.bvmul(
                    A_val_term_intermediate, X_prev_extended)

                Q_prev_intermediate_expr = yices.bvudiv(
                    product_AXprev, M_lcg_term_intermediate_for_Q)

                # Yices C API extract: bvextract(term_t t, uint32_t low, uint32_t high)
                # Wrapper `yices.extract` should follow this (term, low, high_inclusive)
                if hasattr(yices, 'extract'):
                    Q_prev_actual_expr = yices.extract(
                        Q_prev_intermediate_expr, 0, Q_WIDTH-1)
                else:  # Fallback for bv prefix
                    Q_prev_actual_expr = yices.bvextract(
                        Q_prev_intermediate_expr, 0, Q_WIDTH-1)

                rem_Q_expr = yices.bvurem(
                    Q_prev_actual_expr, mod_target_term_for_Qj_cond)
                eq_Q_expr = yices.bveq(
                    rem_Q_expr, val_q_target_term_for_Qj_cond)
                yices.assert_formula(ctx, eq_Q_expr)

                if hasattr(yices, 'extract'):
                    current_X_term = yices.extract(product_AXprev, 0, N_BITS-1)
                else:
                    current_X_term = yices.bvextract(
                        product_AXprev, 0, N_BITS-1)

                rem_X_expr = yices.bvurem(
                    current_X_term, mod_target_term_for_Xj_cond)
                eq_X_expr = yices.bveq(rem_X_expr, val_target_term_for_Xj_cond)
                yices.assert_formula(ctx, eq_X_expr)

        print(f"  Solver checking for k = {k_val_problem}...")
        status = yices.check_context(ctx, None)

        if status == yices.STATUS_SAT:  # yices.STATUS_SAT is often yices_api.STATUS_SAT if it's from yices_api
            print(f"  SAT for k = {k_val_problem}")
            model = yices.get_model(ctx, 1)

            # Get X0 model value
            # The yices-smt package usually provides yices.get_int_value or yices.get_bitvector_value (returning list of bits)
            found_X0_val = -1
            val_list = None

            if hasattr(yices, 'get_bitvector_value'):  # returns list of ints (0 or 1)
                val_list = yices.get_bitvector_value(
                    model, X0_term)  # MSB first typically
                if val_list is not None:
                    bit_string = "".join(map(str, val_list))
                    found_X0_val = int(bit_string, 2)
            elif hasattr(yices, 'get_int_value'):  # If value fits native integer type
                # This may fail or truncate for N_BITS=48 if it only supports up to 64-bit and underlying API expects it
                try:
                    found_X0_val = yices.get_int_value(model, X0_term)
                except Exception as e_val:
                    print(
                        f"    get_int_value failed for X0: {e_val}, trying string.")

            # Fallback for other representations
            if found_X0_val == -1 and hasattr(yices, 'get_value_as_string'):
                str_val = yices.get_value_as_string(model, X0_term)
                print(f"    X0 (as string from Yices): {str_val}")
                # If it's a binary string like "0b1010..." or hex "0xABCD..."
                try:
                    if str_val.startswith("0b"):
                        found_X0_val = int(str_val[2:], 2)
                    elif str_val.startswith("0x"):
                        found_X0_val = int(str_val[2:], 16)
                    else:  # Assume decimal if no prefix
                        found_X0_val = int(str_val)
                except ValueError:
                    print("    Could not parse string value from get_value_as_string.")

            if found_X0_val != -1:
                print(f"  Example X_0 = {found_X0_val}")
            else:
                print("  Could not extract X0 model value.")

            max_k_found = k_val_problem
            X0_for_max_k = found_X0_val
            if hasattr(yices, 'yices_free_model'):
                yices.yices_free_model(model)  # yices-smt uses yices_
            elif hasattr(yices, 'free_model'):
                yices.free_model(model)

        elif status == yices.STATUS_UNSAT:
            print(f"  UNSAT for k = {k_val_problem}")
            if hasattr(yices, 'yices_free_context'):
                yices.yices_free_context(ctx)
            elif hasattr(yices, 'free_context'):
                yices.free_context(ctx)
            break
        else:
            err_str = ""
            if hasattr(yices, 'yices_error_string'):
                err_str = yices.yices_error_string()
            elif hasattr(yices, 'error_string'):
                err_str = yices.error_string()
            print(
                f"  Yices status code: {status} for k={k_val_problem}. Error: {err_str}")
            if hasattr(yices, 'yices_free_context'):
                yices.yices_free_context(ctx)
            elif hasattr(yices, 'free_context'):
                yices.free_context(ctx)

            break

        if hasattr(yices, 'yices_free_context'):
            yices.yices_free_context(ctx)
        elif hasattr(yices, 'free_context'):
            yices.free_context(ctx)

    # if cfg and hasattr(yices, 'yices_free_config'): yices.yices_free_config(cfg)
    if hasattr(yices, 'yices_exit'):
        yices.yices_exit()
    elif hasattr(yices, 'exit'):
        yices.exit()

    print("\n--- Final Result ---")
    if X0_for_max_k is not None and X0_for_max_k != -1:
        print(f"Największe k: {max_k_found}")
        print(f"Odpowiadająca wartość X_0 (jedna z możliwych): {X0_for_max_k}")
    else:
        if max_k_found == -1:
            print("Nie znaleziono X_0 nawet dla k=0 (lub Yices error).")
        else:
            print(
                f"Rozwiązanie znalezione dla k={max_k_found}, ale wystąpił problem z odczytem wartości X_0.")


if __name__ == "__main__":
    main()
