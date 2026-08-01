async function loadCart() {

    // Show loader first
    document.getElementById("cartBody").innerHTML = `
        <tr>
            <td colspan="11" class="text-center py-4">
                <div class="spinner-border text-primary" role="status"></div>
                <div class="mt-2">Loading cart...</div>
            </td>
        </tr>
    `;

    try {

        const res = await fetch("/api/cart-list");
        const data = await res.json();

        let html = "";
        let totalQty = 0;

        data.forEach(item => {

            totalQty += item.qty;

            html += `
<tr
    data-style-no="${item.style_no}"
    data-qty="${item.qty}"
    data-gold-color="${item.gold_color || ''}"
    data-gold-purity="${item.gold_purity || ''}"
    data-diamond-color="${item.diamond_color || ''}"
    data-diamond-clarity="${item.diamond_clarity || ''}"
    data-remarks="${item.remarks || ''}"
>

<td class="text-center">
    <img
        src="data:image/jpeg;base64,${item.image}"
        class="img-thumbnail"
        style="width:90px;height:70px;object-fit:cover;">
</td>

<td><strong>${item.style_no}</strong></td>

<td class="text-center">${item.qty}</td>

<td>${item.gold_color}</td>

<td>${item.gold_purity}</td>

<td>${item.diamond_color}</td>

<td>${item.diamond_clarity}</td>

<td>${item.remarks || "-"}</td>

<td>${item.created_at}</td>

<td class="text-center">
    <button
        type="button"
        class="btn btn-warning"
        onclick="editCart(${item.id})">
        <i class="fas fa-edit"></i>
    </button>
</td>

<td class="text-center">
    <button
        class="btn btn-sm btn-danger"
        onclick="deleteCart(${item.id})">
        <i class="fas fa-trash"></i>
    </button>
</td>

</tr>
`;

        });

        if (html === "") {
            html = `
                <tr>
                    <td colspan="11" class="text-center">
                        Cart is empty.
                    </td>
                </tr>
            `;
        }

        document.getElementById("cartBody").innerHTML = html;

    } catch (err) {

        document.getElementById("cartBody").innerHTML = `
            <tr>
                <td colspan="11" class="text-center text-danger">
                    Failed to load cart.
                </td>
            </tr>
        `;

        Swal.fire({
            icon: "error",
            title: "Error",
            text: "Failed to load cart."
        });

    }

}

loadCart();