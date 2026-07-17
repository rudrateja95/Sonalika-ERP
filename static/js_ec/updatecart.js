// Update Cart
async function updateCart() {

    const id = document.getElementById("edit_cart_id").value;

    const payload = {

        qty: document.querySelector('[data-role="edit_qty"]').value,
        gold_color: document.querySelector('[data-role="edit_gold_color"]').value,
        gold_purity: document.querySelector('[data-role="edit_gold_purity"]').value,
        diamond_color: document.querySelector('[data-role="edit_diamond_color"]').value,
        diamond_clarity: document.querySelector('[data-role="edit_diamond_clarity"]').value,
        remarks: document.querySelector('[data-role="edit_remarks"]').value

    };

    const res = await fetch(`/api/cart/${id}`, {

        method: "PUT",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify(payload)

    });

    const result = await res.json();

    if (result.status == "success") {

        Swal.fire({
            icon: "success",
            title: "Updated",
            text: result.message,
            timer: 1500,
            showConfirmButton: false
        }).then(() => {

            $("#editCartModal").modal("hide");

            loadCart();

        });

    }

}