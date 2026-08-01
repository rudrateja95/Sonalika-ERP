// Open Edit Modal
async function editCart(id) {

    document.getElementById("edit_cart_id").value = id;

    const res = await fetch(`/api/cart/${id}`);

    const item = await res.json();

    document.querySelector('[data-role="edit_style_no"]').value = item.style_no;
    document.querySelector('[data-role="edit_qty"]').value = item.qty;
    document.querySelector('[data-role="edit_gold_color"]').value = item.gold_color;
    document.querySelector('[data-role="edit_gold_purity"]').value = item.gold_purity;
    document.querySelector('[data-role="edit_diamond_color"]').value = item.diamond_color;
    document.querySelector('[data-role="edit_diamond_clarity"]').value = item.diamond_clarity;
    document.querySelector('[data-role="edit_remarks"]').value = item.remarks;

    $("#editCartModal").modal("show");
}