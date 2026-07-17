async function loadCart() {

    const res = await fetch("/api/cart-list");

    const data = await res.json();

    let html = "";

    let totalQty = 0;

    data.forEach(item => {

        totalQty += item.qty;

        html += `
<tr>

<td class="text-center">
    <img
        src="data:image/jpeg;base64,${item.image}"
        class="img-thumbnail"
        style="width:90px;height:70px;object-fit:cover;">
</td>

    <td>
        <strong>${item.style_no}</strong>
    </td>

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

    document.getElementById("cartBody").innerHTML = html;

    document.getElementById("cartCount").innerHTML = data.length;


}

loadCart();